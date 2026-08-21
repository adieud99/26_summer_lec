package com.lecture.rag.chatbotwiki;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.annotation.PostConstruct;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.context.annotation.Profile;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import java.util.List;

@RestController
@Profile("chatbot-api")
@RequestMapping("/api/chatbot")
@Tag(name = "RAG 챗봇 (제주 / 김치)", description = "제주도 위키는 도구(@Tool)로, 김치 위키는 QuestionAnswerAdvisor로 검색하는 RAG 챗봇. 문서에 근거가 없으면 모른다고 답한다.")
public class RagChatbotApiController {

    private static final double DEFAULT_THRESHOLD = 0.5;
    private static final int TOP_K = 3;

    private static final String SYSTEM_PROMPT = """
            항상 한국어로 답변하세요.
            당신은 '제주도 위키'와 '김치 위키' 두 문서에 대해서만 답할 수 있는 문서 기반 도우미입니다.

            규칙:
            1. 제주도에 관한 질문이면 반드시 searchJejuWiki 도구를 호출하고, 그 검색 결과만 근거로 답하세요.
            2. 김치에 관한 질문이면 함께 제공된 컨텍스트(검색된 문서)만 근거로 답하세요.
            3. 도구 결과나 컨텍스트에 답의 근거가 없으면 절대 추측하지 말고
               "제가 가진 문서에는 그 내용이 없어서 모르겠습니다." 라고만 답하세요.
            4. 문서에 없는 사실을 일반 상식으로 채워 넣지 마세요.
            """;

    private final ChatModel chatModel;
    private final VectorStore vectorStore;
    private final WikiIndexer indexer;

    private ChatClient chatClient;

    public RagChatbotApiController(ChatModel chatModel, VectorStore vectorStore, WikiIndexer indexer) {
        this.chatModel = chatModel;
        this.vectorStore = vectorStore;
        this.indexer = indexer;
    }

    @PostConstruct
    void init() {
        indexer.indexIfAbsent();
        this.chatClient = ChatClient.builder(chatModel)
                .defaultSystem(SYSTEM_PROMPT)
                .build();
    }

    public record AskResult(String question, int searchHitCount, List<String> hitSources,
                            boolean jejuToolCalled, String answer) {}

    public record SearchHit(String source, String preview) {}

    @Operation(summary = "RAG 챗봇에게 질문하기",
            description = "제주도 질문이면 searchJejuWiki 도구가 호출되고, 김치 질문이면 QuestionAnswerAdvisor가 컨텍스트를 붙인다. "
                    + "두 문서 어디에도 임계값을 넘는 청크가 없으면 LLM을 아예 호출하지 않고 '모르겠습니다'를 반환한다. "
                    + "응답의 jejuToolCalled / hitSources 필드로 어느 경로를 탔는지 확인할 수 있다.")
    @GetMapping("/ask")
    public AskResult ask(
            @Parameter(description = "질문. 제주도 예) 제주도 면적이 얼마야? / 김치 예) 김치는 언제 Codex에 등록됐어? / 문서에 없는 예) 아이폰 17 가격 알려줘",
                    example = "제주도 면적이 얼마야?")
            @RequestParam(defaultValue = "제주도 면적이 얼마야?") String question,

            @Parameter(description = "코사인 유사도 임계값. 낮추면 잘 답하지만 헛소리가 늘고, 높이면 '모른다'가 늘어난다.",
                    example = "0.5")
            @RequestParam(defaultValue = "0.5") double threshold) {

        List<Document> hits = vectorStore.similaritySearch(
                SearchRequest.builder()
                        .query(question)
                        .topK(TOP_K)
                        .similarityThreshold(threshold)
                        .build());

        if (hits.isEmpty()) {
            return new AskResult(question, 0, List.of(), false,
                    "제가 가진 문서에는 그 내용이 없어서 모르겠습니다.");
        }

        List<String> sources = hits.stream()
                .map(d -> String.valueOf(d.getMetadata().get("source")))
                .distinct()
                .toList();

        QuestionAnswerAdvisor kimchiAdvisor = QuestionAnswerAdvisor.builder(vectorStore)
                .searchRequest(SearchRequest.builder()
                        .filterExpression("source == '" + WikiIndexer.SOURCE_KIMCHI + "'")
                        .topK(TOP_K)
                        .similarityThreshold(threshold)
                        .build())
                .build();

        JejuWikiTool jejuTool = new JejuWikiTool(vectorStore, TOP_K, threshold);

        String answer = chatClient.prompt()
                .advisors(kimchiAdvisor)
                .tools(jejuTool)
                .user(question)
                .call()
                .content();

        return new AskResult(question, hits.size(), sources, jejuTool.getCallCount() > 0, answer);
    }

    @Operation(summary = "유사도 검색 결과만 보기 (LLM 호출 없음)",
            description = "'모른다'가 어떻게 결정되는지 눈으로 확인하는 용도. LLM을 거치지 않고 PGVector 검색 결과만 돌려준다. "
                    + "여기서 빈 배열이 나오는 질문은 /ask에서도 '모르겠습니다'가 나온다. "
                    + "source를 지정하면 그 문서 안에서만 검색한다.")
    @GetMapping("/search")
    public List<SearchHit> search(
            @Parameter(description = "검색할 질문 또는 키워드", example = "올레길 몇 코스야?")
            @RequestParam(defaultValue = "올레길 몇 코스야?") String question,

            @Parameter(description = "코사인 유사도 임계값", example = "0.5")
            @RequestParam(defaultValue = "0.5") double threshold,

            @Parameter(description = "(선택) 특정 문서로 한정. jeju-wiki 또는 kimchi-wiki. 비우면 두 문서 전체.")
            @RequestParam(required = false) String source) {

        SearchRequest.Builder builder = SearchRequest.builder()
                .query(question)
                .topK(TOP_K)
                .similarityThreshold(threshold);

        if (source != null && !source.isBlank()) {
            builder.filterExpression("source == '" + source + "'");
        }

        return vectorStore.similaritySearch(builder.build()).stream()
                .map(doc -> {
                    String text = doc.getText().replaceAll("\\s+", " ");
                    return new SearchHit(
                            String.valueOf(doc.getMetadata().get("source")),
                            text.substring(0, Math.min(150, text.length())));
                })
                .toList();
    }
}