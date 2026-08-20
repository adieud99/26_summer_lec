package com.lecture.rag.chatbotwiki;

import org.springframework.ai.document.Document;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import java.util.List;

public class JejuWikiTool {

    private final VectorStore vectorStore;
    private final int topK;
    private final double similarityThreshold;
    private int callCount = 0;

    public JejuWikiTool(VectorStore vectorStore, int topK, double similarityThreshold) {
        this.vectorStore = vectorStore;
        this.topK = topK;
        this.similarityThreshold = similarityThreshold;
    }

    public int getCallCount() {
        return callCount;
    }

    @Tool(description = "제주도 위키백과 문서에서 질문과 관련된 내용을 검색한다. "
            + "제주도의 면적, 인구, 지리, 기후, 강수량, 형성 과정과 화산활동, 올레길·성산일출봉 같은 관광지 등 "
            + "제주도에 관한 질문에만 사용할 것. 김치나 그 밖의 주제에는 사용하지 말 것.")
    public String searchJejuWiki(
            @ToolParam(description = "제주도 위키 문서에서 검색할 질문 또는 키워드") String query) {

        callCount++;
        System.out.println("  >>> [도구 호출됨] searchJejuWiki(\"" + query + "\")");

        List<Document> results = vectorStore.similaritySearch(
                SearchRequest.builder()
                        .query(query)
                        .filterExpression("source == '" + WikiIndexer.SOURCE_JEJU + "'")
                        .topK(topK)
                        .similarityThreshold(similarityThreshold)
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
}