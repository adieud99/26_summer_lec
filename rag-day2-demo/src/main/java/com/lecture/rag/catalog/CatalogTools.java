package com.lecture.rag.catalog;

import org.springframework.ai.document.Document;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import java.util.List;

public class CatalogTools {

    private static final int TOP_K = 4;
    private static final double SIMILARITY_THRESHOLD = 0.5;

    private final VectorStore vectorStore;
    private final DocumentCatalog catalog;

    private int catalogCallCount = 0;
    private int searchCallCount = 0;

    public CatalogTools(VectorStore vectorStore, DocumentCatalog catalog) {
        this.vectorStore = vectorStore;
        this.catalog = catalog;
    }

    public int getCatalogCallCount() { return catalogCallCount; }
    public int getSearchCallCount() { return searchCallCount; }

    @Tool(description = """
            보유한 문서의 목록·개수·카테고리를 조회한다. 문서의 '내용'이 아니라 '어떤 자료를 몇 개 갖고 있는지'를
            묻는 질문에 사용할 것. 예: "자료 몇 개 있어?", "어떤 문서들 갖고 있어?", "위키 문서 몇 개야?",
            "약관 문서 있어?", "무슨 카테고리가 있어?".
            문서 내용을 묻는 질문에는 사용하지 말 것.
            """)
    public String browseCatalog(
            @ToolParam(required = false,
                    description = "특정 카테고리만 보고 싶을 때 지정. 가능한 값: 위키, 약관, 매뉴얼, 논문, 오픈소스. 전체를 보려면 비워둘 것.")
            String category) {

        catalogCallCount++;
        System.out.println("  >>> [도구 호출] browseCatalog(category=" + category + ")  ← SQL");

        List<DocumentCatalog.DocumentInfo> docs = (category == null || category.isBlank())
                ? catalog.findAll()
                : catalog.findByCategory(category);

        if (docs.isEmpty()) {
            return "해당 조건에 맞는 문서가 없습니다. 보유 카테고리: " + catalog.distinctCategories();
        }

        StringBuilder sb = new StringBuilder();
        sb.append("총 ").append(docs.size()).append("개 문서");
        if (category != null && !category.isBlank()) {
            sb.append(" (카테고리: ").append(category).append(")");
        }
        sb.append("\n");
        for (DocumentCatalog.DocumentInfo d : docs) {
            sb.append("- ").append(d.title())
                    .append(" [").append(d.category()).append("]")
                    .append(" (청크 ").append(d.chunkCount()).append("개, id=").append(d.documentId()).append(")\n");
        }
        return sb.toString();
    }

    @Tool(description = """
            문서의 '내용'을 검색한다. 사실 관계를 묻는 질문에 사용할 것.
            예: "제주도 면적이 얼마야?", "환불 조항 찾아줘", "김치는 언제 Codex에 등록됐어?".
            category를 지정하면 그 카테고리 문서 안에서만 검색한다.
            문서 개수나 목록을 묻는 질문에는 사용하지 말 것.
            """)
    public String searchDocuments(
            @ToolParam(description = "검색할 질문 또는 키워드") String query,
            @ToolParam(required = false,
                    description = "검색 범위를 좁힐 카테고리. 가능한 값: 위키, 약관, 매뉴얼, 논문, 오픈소스. 전체 검색이면 비워둘 것.")
            String category) {

        searchCallCount++;
        System.out.println("  >>> [도구 호출] searchDocuments(query=\"" + query + "\", category=" + category + ")  ← 벡터");

        List<String> ids;
        if (category == null || category.isBlank()) {
            ids = catalog.allDocumentIds();
        } else {
            ids = catalog.findDocumentIdsByCategory(category);
            System.out.println("      SQL로 좁힌 문서: " + ids);
            if (ids.isEmpty()) {
                return "'" + category + "' 카테고리 문서가 없습니다. 보유 카테고리: " + catalog.distinctCategories();
            }
        }

        String filter = "document_id in [" + ids.stream()
                .map(id -> "'" + id + "'")
                .reduce((a, b) -> a + ", " + b)
                .orElse("''") + "]";

        List<Document> hits = vectorStore.similaritySearch(
                SearchRequest.builder()
                        .query(query)
                        .filterExpression(filter)
                        .topK(TOP_K)
                        .similarityThreshold(SIMILARITY_THRESHOLD)
                        .build());

        if (hits.isEmpty()) {
            return "관련 내용을 문서에서 찾지 못했습니다. 모른다고 답하세요.";
        }

        StringBuilder sb = new StringBuilder();
        for (Document doc : hits) {
            String docId = String.valueOf(doc.getMetadata().get("document_id"));
            DocumentCatalog.DocumentInfo info = catalog.findById(docId);
            String label = (info == null) ? docId : info.title() + " [" + info.category() + "]";

            sb.append("- (출처: ").append(label).append(") ")
                    .append(doc.getText().replaceAll("\\s+", " ")).append("\n");
        }
        return sb.toString();
    }
}