package com.lecture.rag.catalog;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.annotation.PostConstruct;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.context.annotation.Profile;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import java.util.List;

@RestController
@Profile("catalog-api")
@RequestMapping("/api/catalog")
@Tag(name = "문서 사서 챗봇 (카탈로그 + 벡터 하이브리드)",
        description = "문서 목록/개수 질문은 SQL 카탈로그로, 내용 질문은 벡터 검색으로 — 모델이 도구를 골라서 답한다")
public class HybridChatbotApiController {

    private static final String SYSTEM_PROMPT = """
            항상 한국어로 답변하세요.
            당신은 문서 저장소를 관리하는 사서입니다. 도구 두 개를 쓸 수 있습니다.

            1. browseCatalog — 보유 문서의 목록·개수·카테고리를 조회합니다.
            2. searchDocuments — 문서의 내용을 검색합니다. 카테고리로 좁힐 수 있습니다.

            규칙:
            - 개수나 목록을 묻는 질문에 벡터 검색을 쓰지 마세요. 셀 수 없습니다.
            - 도구 결과에 근거가 없으면 추측하지 말고
              "제가 가진 문서에는 그 내용이 없어서 모르겠습니다." 라고만 답하세요.
            - 내용을 답할 때는 출처(문서 제목)를 함께 밝히세요.
            """;

    private final ChatModel chatModel;
    private final CatalogIndexer indexer;
    private final CatalogTools tools;
    private final DocumentCatalog catalog;

    private ChatClient chatClient;

    public HybridChatbotApiController(ChatModel chatModel, CatalogIndexer indexer,
                                      CatalogTools tools, DocumentCatalog catalog) {
        this.chatModel = chatModel;
        this.indexer = indexer;
        this.tools = tools;
        this.catalog = catalog;
    }

    @PostConstruct
    void init() {
        indexer.indexIfAbsent();
        this.chatClient = ChatClient.builder(chatModel)
                .defaultSystem(SYSTEM_PROMPT)
                .build();
    }

    public record AskResult(String question, int catalogToolCalls, int vectorToolCalls, String answer) {}

    @Operation(summary = "사서에게 질문하기",
            description = "질문 성격에 따라 모델이 도구를 고른다. 응답의 catalogToolCalls / vectorToolCalls로 "
                    + "어느 경로를 탔는지 확인할 수 있다. "
                    + "목록·개수 질문 예) 자료 몇 개 갖고 있어? / 위키 문서 몇 개야? "
                    + "내용 질문 예) 제주도 면적이 얼마야? / 약관 문서에서 환불 조항 찾아줘")
    @GetMapping("/ask")
    public AskResult ask(
            @Parameter(description = "질문", example = "위키 문서 몇 개 갖고 있어?")
            @RequestParam(defaultValue = "위키 문서 몇 개 갖고 있어?") String question) {

        int catalogBefore = tools.getCatalogCallCount();
        int searchBefore = tools.getSearchCallCount();

        String answer = chatClient.prompt()
                .tools(tools)
                .user(question)
                .call()
                .content();

        return new AskResult(question,
                tools.getCatalogCallCount() - catalogBefore,
                tools.getSearchCallCount() - searchBefore,
                answer);
    }

    @Operation(summary = "카탈로그 원본 보기 (LLM 없음)",
            description = "document_catalog 테이블을 그대로 조회한다. 벡터가 아니라 SQL이 답하는 영역이라는 걸 확인하는 용도.")
    @GetMapping("/documents")
    public List<DocumentCatalog.DocumentInfo> documents(
            @Parameter(description = "(선택) 카테고리 필터. 위키 / 약관 / 매뉴얼 / 논문 / 오픈소스")
            @RequestParam(required = false) String category) {
        return (category == null || category.isBlank())
                ? catalog.findAll()
                : catalog.findByCategory(category);
    }
}