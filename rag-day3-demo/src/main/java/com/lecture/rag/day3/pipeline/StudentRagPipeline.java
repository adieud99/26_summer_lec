package com.lecture.rag.day3.pipeline;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.document.Document;
import org.springframework.stereotype.Component;

import com.lecture.rag.day3.agent.ChatRequest;
import com.lecture.rag.day3.agent.RagOptions;
import com.lecture.rag.day3.agent.SourceRef;
import com.lecture.rag.day3.knowledge.KnowledgeBase;

/**
 * ★ 캡스톤 실습 파일 ★ — 네 개의 훅을 모두 구현한 버전.
 *
 * <pre>
 *  실버 ① rewriteQueries()  이전 대화 + 질문 → 검색용 질문 3개(+원문) multi-query
 *  실버 ② keywordSearch()   전체 청크를 단어 매칭으로 채점하는 하이브리드 검색
 *  실버 ③ rerank()          Day2 Lab2.2 LlmReranker 방식(0~10점 채점) 그대로
 *  골드   selfCheck()       답변이 근거 안에 있는지 LLM에게 재확인 (Self-RAG 축소판)
 * </pre>
 *
 * <p>수업에서 쓴 API만 사용했다 — {@code chatClient().prompt()...call().content()},
 * {@code knowledgeBase.chunksOf()}, {@code RagPrompts}의 헬퍼들. 별도 라이브러리는 추가하지 않았다.
 */
@Component
public class StudentRagPipeline extends AbstractRagPipeline {

    /** LLM이 "숫자만" 지시를 안 지키고 설명을 붙일 때를 대비해 응답에서 숫자를 훑는다 (Day2 LlmReranker 변형). */
    private static final Pattern NUMBER = Pattern.compile("\\d+");

    /** 재작성 쿼리는 원문 포함 최대 4개까지만 — 쿼리 하나당 검색이 한 번씩 더 나간다. */
    private static final int MAX_QUERIES = 4;

    /** 채점/검색용 내부 점수 홀더. */
    private record Scored(Document doc, int score) {
    }

    public StudentRagPipeline(KnowledgeBase knowledgeBase, ChatModel chatModel) {
        super(knowledgeBase, chatModel);
    }

    @Override
    public String id() {
        return "student";
    }

    @Override
    public String name() {
        return "내 파이프라인";
    }

    @Override
    public String tier() {
        // 골드 과제(selfCheck)까지 구현해서 gold로 올렸다. 실버만 구현했다면 "silver"로 되돌릴 것.
        return "gold";
    }

    @Override
    public String description() {
        return "질문 재작성 + 키워드 하이브리드 검색 + LLM 재정렬 + 자기 검증까지 붙인 파이프라인.";
    }

    @Override
    public List<String> supportedFeatures() {
        return List.of(
                RagOptions.FEATURE_REWRITE,
                RagOptions.FEATURE_KEYWORD,
                RagOptions.FEATURE_RERANK,
                RagOptions.FEATURE_SELF_CHECK);
    }

    // =================================================================== 실버 ①

    /**
     * 질문 재작성 / multi-query 검색.
     *
     * <p>짧고 모호한 질문("그거 얼마야?")은 임베딩이 잡을 정보가 거의 없다. 이전 대화를 참고해
     * 완전한 문장으로 다시 쓰고, 표현이 다른 여러 버전을 만들어 각각 검색하면 회수율(recall)이 올라간다.
     *
     * <p>원문 질문을 목록 맨 앞에 그대로 남긴다 — 재작성이 엉뚱하게 나와도 최소한의 검색 품질은 보장된다.
     */
    @Override
    protected Optional<List<String>> rewriteQueries(String question, List<ChatRequest.Turn> history,
            RagOptions options) {
        if (question == null || question.isBlank()) {
            return Optional.empty();
        }

        String historyText = RagPrompts.historyAsText(history, options.maxHistoryOrDefault());
        String prompt = """
                [이전 대화]
                %s

                [마지막 질문]
                %s

                위 마지막 질문을 문서 검색에 쓸 수 있게 완전한 문장으로 다시 쓰세요.
                서로 표현이 다른 3개를 줄바꿈으로만 구분해서 한국어로 출력하세요.
                번호, 설명, 따옴표는 절대 붙이지 마세요.
                """.formatted(historyText.isBlank() ? "(없음)" : historyText, question);

        // 로컬 소형 모델(llama3.2:3b)은 지시를 잘 안 지켜서 시스템 메시지로 한 번 더 못 박는다 (Day2 M2.5와 같은 이유)
        String response = chatClient().prompt()
                .options(ChatOptions.builder().temperature(0.0))
                .system("재작성된 검색어만 한 줄에 하나씩 한국어로 출력하세요. 설명, 번역, 번호는 절대 쓰지 마세요.")
                .user(prompt)
                .call()
                .content();

        List<String> queries = new ArrayList<>();
        queries.add(question.strip());
        if (response != null) {
            for (String line : response.split("\\R")) {
                // 모델이 "1. ", "- " 같은 머리표를 붙이는 경우가 많아 잘라낸다
                String cleaned = line.replaceAll("^[\\s\\-*·•]*\\d*[.)]?\\s*", "").strip();
                if (cleaned.length() >= 2 && !queries.contains(cleaned)) {
                    queries.add(cleaned);
                }
            }
        }
        return Optional.of(queries.size() > MAX_QUERIES ? queries.subList(0, MAX_QUERIES) : queries);
    }

    // =================================================================== 실버 ②

    /**
     * 키워드 검색(하이브리드 검색의 절반).
     *
     * <p>벡터 검색은 "의미가 비슷한 것"을 찾기 때문에 모델명(RTX-4090)·조항 번호(제12조)처럼
     * <b>정확히 일치해야 하는 것</b>에 약하다. 질문 단어가 몇 개나 등장하는지로 청크를 채점해서
     * 벡터 검색 결과에 합친다(합치는 건 AbstractRagPipeline이 해준다).
     *
     * <p>점수가 0인 청크는 버린다 — 안 버리면 관련 없는 청크가 컨텍스트를 오염시킨다.
     */
    @Override
    protected Optional<List<Document>> keywordSearch(String query, List<String> docIds, RagOptions options) {
        List<String> words = extractKeywords(query);
        if (words.isEmpty()) {
            return Optional.of(List.of());
        }

        List<Scored> scored = new ArrayList<>();
        for (Document chunk : this.knowledgeBase.chunksOf(docIds)) {
            String text = chunk.getText() == null ? "" : chunk.getText().toLowerCase();
            int matched = 0;
            for (String word : words) {
                if (text.contains(word)) {
                    matched++;
                }
            }
            if (matched > 0) {
                scored.add(new Scored(chunk, matched));
            }
        }

        return Optional.of(scored.stream()
                .sorted(Comparator.comparingInt(Scored::score).reversed())
                .limit(options.topKOrDefault())
                .map(Scored::doc)
                .toList());
    }

    /**
     * 질문에서 검색에 쓸 단어만 남긴다. 공백으로 쪼개고, 기호를 떼고, 2글자 미만은 버린다.
     *
     * <p>한계: 한국어는 조사가 붙어("제주도의") 본문의 "제주도"와 매칭이 안 될 수 있다.
     * 더 해보고 싶으면 흔한 단어의 가중치를 낮추는 BM25(Day2 M2.4)로 확장하면 된다.
     */
    private List<String> extractKeywords(String query) {
        List<String> words = new ArrayList<>();
        if (query == null || query.isBlank()) {
            return words;
        }
        for (String token : query.toLowerCase().split("\\s+")) {
            String word = token.replaceAll("[^0-9a-z가-힣]", "");
            if (word.length() >= 2 && !words.contains(word)) {
                words.add(word);
            }
        }
        return words;
    }

    // =================================================================== 실버 ③

    /**
     * 재정렬(rerank) — Day2 Lab2.2의 {@code LlmReranker}를 이 파이프라인에 맞게 옮겨온 것.
     *
     * <p>임베딩 유사도 1위가 항상 정답 청크는 아니다. rerank 토글이 켜지면 검색을 topK의 2배로 넓게
     * 해오므로(AbstractRagPipeline), 그 후보를 LLM에게 0~10점으로 채점시켜 상위 topK개만 남긴다.
     *
     * <p>주의: 후보 8개면 LLM을 8번 부른다 → 답변까지 10초 이상 걸릴 수 있다. 느린 게 정상이고,
     * 그 대가로 정확도를 사는 것이다.
     */
    @Override
    protected Optional<List<Document>> rerank(String query, List<Document> candidates, RagOptions options) {
        if (candidates == null || candidates.isEmpty()) {
            return Optional.of(List.of());
        }

        List<Scored> scored = new ArrayList<>(candidates.size());
        for (Document candidate : candidates) {
            scored.add(new Scored(candidate, scoreRelevance(query, candidate)));
        }

        return Optional.of(scored.stream()
                .sorted(Comparator.comparingInt(Scored::score).reversed())
                .limit(options.topKOrDefault())
                .map(Scored::doc)
                .toList());
    }

    /** 청크 하나를 0~10점으로 채점. 응답에서 첫 숫자만 뽑아 군더더기를 방어한다. */
    private int scoreRelevance(String query, Document document) {
        String prompt = """
                질문: %s
                문서: %s
                이 문서가 질문에 답하는 데 얼마나 관련 있는지 0~10 사이 숫자 하나만 답하세요. 설명은 하지 마세요.
                """.formatted(query, RagPrompts.squeeze(document.getText()));

        String response = chatClient().prompt()
                .options(ChatOptions.builder().temperature(0.0))
                .system("항상 숫자만 답하세요.")
                .user(prompt)
                .call()
                .content();

        if (response == null) {
            return 0;
        }
        // 첫 숫자를 그냥 쓰면 "2024년 기준으로 관련 없음" 같은 응답이 만점이 되어버린다.
        // 0~10 범위에 드는 첫 숫자만 점수로 인정하고, 없으면 0점 처리한다.
        Matcher matcher = NUMBER.matcher(response);
        while (matcher.find()) {
            String number = matcher.group();
            if (number.length() <= 2) {
                int value = Integer.parseInt(number);
                if (value <= 10) {
                    return value;
                }
            }
        }
        return 0;
    }

    // =================================================================== 골드

    /**
     * 자기 검증(self-check) — Self-RAG의 축소판.
     *
     * <p>RAG를 붙여도 LLM은 컨텍스트에 없는 내용을 슬쩍 섞는다. 생성이 끝난 답변을 다시 한 번
     * "근거 안에 실제로 있는 내용인가"로 검사해서, 없으면 화면에 경고를 띄운다.
     *
     * <p>형식을 JSON이 아니라 "통과 / 주의: ..." 한 줄로 요구하는 이유는 3B급 로컬 모델이 JSON을
     * 자주 깨먹기 때문이다(JudgeService와 같은 이유).
     */
    @Override
    protected Optional<String> selfCheck(String question, String answer, List<SourceRef> sources,
            RagOptions options) {
        if (answer == null || answer.isBlank()) {
            return Optional.of("주의: 생성된 답변이 없어 검증할 수 없습니다.");
        }
        if (sources == null || sources.isEmpty()) {
            return Optional.of("주의: 근거 청크가 하나도 없습니다. 답변 전체가 검증되지 않았습니다.");
        }

        String prompt = """
                [근거]
                %s

                [질문]
                %s

                [답변]
                %s

                답변의 모든 문장이 [근거] 안에 실제로 있는 내용인지 판정하세요.
                근거에 없는 내용이 하나도 없으면 '통과' 한 단어만 출력하세요.
                하나라도 있으면 '주의: <근거에 없는 내용 한 문장 요약>' 형식으로 한 줄만 출력하세요.
                다른 말은 절대 붙이지 마세요.
                """.formatted(RagPrompts.formatContext(sources), question, answer);

        // 판정은 창의성이 필요 없다 — 온도를 0으로 (JudgeService와 동일)
        String raw = chatClient().prompt()
                .options(ChatOptions.builder().temperature(0.0))
                .system("판정 결과 한 줄만 한국어로 출력하세요.")
                .user(prompt)
                .call()
                .content();

        if (raw == null || raw.isBlank()) {
            return Optional.of("판정 실패: 검증 모델이 응답하지 않았습니다.");
        }

        String verdict = raw.strip().lines()
                .map(String::strip)
                .filter(line -> !line.isEmpty())
                .findFirst()
                .orElse("");

        if (verdict.startsWith("주의")) {
            return Optional.of(shorten(verdict));
        }
        if (verdict.contains("통과")) {
            return Optional.of("통과 — 답변 내용이 근거 청크 " + sources.size() + "개 범위 안에 있습니다.");
        }
        // 형식을 안 지킨 경우: 원문을 그대로 보여줘서 화면에서 확인할 수 있게 한다
        return Optional.of("판정 형식 이탈 — 원문: " + shorten(verdict));
    }

    /** 화면 단계 카드 한 줄에 들어가도록 너무 긴 판정문을 자른다. */
    private String shorten(String text) {
        return text.length() > 160 ? text.substring(0, 160) + "…" : text;
    }
}
