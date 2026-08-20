package com.lecture.rag.catalog;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;
import java.util.Scanner;

@Component
@Profile("catalog")
public class HybridRagChatbot implements CommandLineRunner {

    private static final String SYSTEM_PROMPT = """
            항상 한국어로 답변하세요.
            당신은 문서 저장소를 관리하는 사서입니다. 도구 두 개를 쓸 수 있습니다.

            1. browseCatalog — 보유 문서의 목록·개수·카테고리를 조회합니다.
               "자료 몇 개 있어?", "어떤 문서 갖고 있어?", "위키 문서 몇 개야?" 같은 질문에 사용하세요.
            2. searchDocuments — 문서의 내용을 검색합니다.
               "제주도 면적이 얼마야?", "환불 조항 찾아줘" 같은 질문에 사용하세요.
               특정 카테고리로 좁히라는 요청이면 category 인자를 함께 넘기세요.

            규칙:
            - 개수나 목록을 묻는 질문에 벡터 검색(searchDocuments)을 쓰지 마세요. 셀 수 없습니다.
            - 도구 결과에 근거가 없으면 추측하지 말고
              "제가 가진 문서에는 그 내용이 없어서 모르겠습니다." 라고만 답하세요.
            - 내용을 답할 때는 도구가 알려준 출처(문서 제목)를 함께 밝히세요.
            """;

    private final ChatModel chatModel;
    private final CatalogIndexer indexer;
    private final CatalogTools tools;

    public HybridRagChatbot(ChatModel chatModel, CatalogIndexer indexer, CatalogTools tools) {
        this.chatModel = chatModel;
        this.indexer = indexer;
        this.tools = tools;
    }

    @Override
    public void run(String... args) {
        indexer.indexIfAbsent();

        ChatClient chatClient = ChatClient.builder(chatModel)
                .defaultSystem(SYSTEM_PROMPT)
                .build();

        System.out.println("=== 문서 사서 챗봇 준비 완료 ===");
        System.out.println("예시) 자료 몇 개 갖고 있어? / 위키 문서 몇 개야? / 제주도 면적이 얼마야?");
        System.out.println("      약관 문서에서 환불 조항 찾아줘 / 아이폰 17 가격 알려줘");
        System.out.println("=== 종료하려면 빈 줄 입력 ===");
        System.out.println();

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("질문> ");
            if (!scanner.hasNextLine()) break;
            String question = scanner.nextLine();
            if (question == null || question.isBlank()) break;

            int catalogBefore = tools.getCatalogCallCount();
            int searchBefore = tools.getSearchCallCount();

            String answer = chatClient.prompt()
                    .advisors(SimpleLoggerAdvisor.builder().build())
                    .tools(tools)
                    .user(question)
                    .call()
                    .content();

            System.out.println("  (경로: 카탈로그 " + (tools.getCatalogCallCount() - catalogBefore)
                    + "회 / 벡터 " + (tools.getSearchCallCount() - searchBefore) + "회)");
            System.out.println("답변> " + answer);
            System.out.println();
        }

        System.out.println("=== 종료 ===");
    }
}