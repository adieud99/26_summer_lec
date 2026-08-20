package com.lecture.rag.chatbotwiki;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.document.Document;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

@Component
@Profile("chatbot")
public class RagCliChatbot implements CommandLineRunner {

    /** 코사인 유사도 임계값 */
    private static final double SIMILARITY_THRESHOLD = 0.5;
    private static final int TOP_K = 3;

    private static final String SOURCE_JEJU = "jeju-wiki";
    private static final String SOURCE_KIMCHI = "kimchi-wiki";

    private static final String JEJU_PATH = "scenarios/6-wiki-jeju.txt";
    private static final String KIMCHI_PATH = "scenarios/7-wiki-kimchi.txt";

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
    private final VectorStore vectorStore; // PgVectorStore가 자동 주입됨
    private int toolCallCount = 0;

    public RagCliChatbot(ChatModel chatModel, VectorStore vectorStore) {
        this.chatModel = chatModel;
        this.vectorStore = vectorStore;
    }

    @Override
    public void run(String... args) {
        indexIfAbsent(JEJU_PATH, SOURCE_JEJU);
        indexIfAbsent(KIMCHI_PATH, SOURCE_KIMCHI);
        System.out.println();

        ChatClient chatClient = ChatClient.builder(chatModel)
                .defaultSystem(SYSTEM_PROMPT)
                .build();

        QuestionAnswerAdvisor kimchiAdvisor = QuestionAnswerAdvisor.builder(vectorStore)
                .searchRequest(SearchRequest.builder()
                        .filterExpression("source == '" + SOURCE_KIMCHI + "'")
                        .topK(TOP_K)
                        .similarityThreshold(SIMILARITY_THRESHOLD)
                        .build())
                .build();

        System.out.println("=== RAG 챗봇 준비 완료 (제주도 / 김치 문서) ===");
        System.out.println("=== 종료하려면 빈 줄 입력 ===");
        System.out.println();

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("질문> ");
            if (!scanner.hasNextLine()) break;
            String question = scanner.nextLine();
            if (question == null || question.isBlank()) break;

            List<Document> hits = vectorStore.similaritySearch(
                    SearchRequest.builder()
                            .query(question)
                            .topK(TOP_K)
                            .similarityThreshold(SIMILARITY_THRESHOLD)
                            .build());

            if (hits.isEmpty()) {
                System.out.println("  (검색 결과 0건 — 임계값 " + SIMILARITY_THRESHOLD + " 미달, LLM 호출 생략)");
                System.out.println("답변> 제가 가진 문서에는 그 내용이 없어서 모르겠습니다.");
                System.out.println();
                continue;
            }
            printHits(hits);

            int before = toolCallCount;
            String answer = chatClient.prompt()
                    .advisors(kimchiAdvisor, SimpleLoggerAdvisor.builder().build())
                    .tools(this)
                    .user(question)
                    .call()
                    .content();

            System.out.println("  (제주 도구 " + (toolCallCount > before ? "호출됨" : "호출 안 됨") + ")");
            System.out.println("답변> " + answer);
            System.out.println();
        }

        System.out.println("=== 종료 ===");
    }

    @Tool(description = "제주도 위키백과 문서에서 질문과 관련된 내용을 검색한다. "
            + "제주도의 면적, 인구, 지리, 기후, 강수량, 형성 과정과 화산활동, 올레길·성산일출봉 같은 관광지 등 "
            + "제주도에 관한 질문에만 사용할 것. 김치나 그 밖의 주제에는 사용하지 말 것.")
    public String searchJejuWiki(
            @ToolParam(description = "제주도 위키 문서에서 검색할 질문 또는 키워드") String query) {

        toolCallCount++;
        System.out.println("  >>> [도구 호출됨] searchJejuWiki(\"" + query + "\")");

        List<Document> results = vectorStore.similaritySearch(
                SearchRequest.builder()
                        .query(query)
                        .filterExpression("source == '" + SOURCE_JEJU + "'")
                        .topK(TOP_K)
                        .similarityThreshold(SIMILARITY_THRESHOLD)
                        .build());

        if (results.isEmpty()) {
            return "제주도 문서에서 관련 내용을 찾지 못했습니다. 모른다고 답하세요.";
        }

        StringBuilder sb = new StringBuilder();
        for (Document doc : results) {
            sb.append("- ").append(doc.getText().replaceAll("\\s+", " ")).append("\n");
        }
        return sb.toString();
    }

    private void indexIfAbsent(String path, String source) {
        boolean exists = !vectorStore.similaritySearch(
                SearchRequest.builder()
                        .query(source)
                        .filterExpression("source == '" + source + "'")
                        .topK(1)
                        .similarityThresholdAll()
                        .build()).isEmpty();

        if (exists) {
            System.out.println("[인덱싱] " + source + " — 이미 적재되어 있어 건너뜀");
            return;
        }

        Document doc = loadText(path, source);

        List<Document> chunks = TokenTextSplitter.builder()
                .withChunkSize(200)   // 위키 문서가 짧아서 200 토큰이면 섹션 단위로 잘 떨어진다
                .build()
                .apply(List.of(doc));

        vectorStore.add(chunks);
        System.out.println("[인덱싱] " + source + " — 청크 " + chunks.size() + "개 적재 완료");
    }

    private Document loadText(String path, String source) {
        try (var in = new ClassPathResource(path).getInputStream()) {
            String text = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            return new Document(text, Map.of("source", source));
        } catch (IOException e) {
            throw new RuntimeException("문서 로드 실패: " + path, e);
        }
    }

    private void printHits(List<Document> hits) {
        System.out.println("  (검색 결과 " + hits.size() + "건)");
        for (Document doc : hits) {
            String preview = doc.getText().replaceAll("\\s+", " ");
            preview = preview.substring(0, Math.min(70, preview.length()));
            System.out.println("   - [" + doc.getMetadata().get("source") + "] " + preview + "...");
        }
    }
}