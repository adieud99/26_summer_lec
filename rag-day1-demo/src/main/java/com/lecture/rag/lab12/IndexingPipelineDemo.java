package com.lecture.rag.lab12;

import org.springframework.ai.document.Document;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Lab 1.2 — 문서 로딩 -> 청킹 -> VectorStore 인덱싱 -> 검색
 * 실행: ./mvnw spring-boot:run -Dspring-boot.run.profiles=lab12
 * 실습 문서: 실제 arXiv 논문 (2026-03, arXiv:2603.07379, 25페이지)
 *
 * [Lab 2.1 적용] SimpleVectorStore(인메모리) → PgVectorStore(도커 PGVector)로 교체.
 *
 * 실제로 바뀐 것은 3줄뿐이다.
 *   1) EmbeddingModel 주입 → VectorStore 주입 (필드/생성자)
 *   2) SimpleVectorStore.builder(...).build() 삭제 — 더 이상 직접 생성하지 않음
 *   3) SimpleVectorStore import 제거
 *
 * vectorStore.add() 와 vectorStore.similaritySearch() 호출부는 한 글자도 안 바뀐다.
 * VectorStore 가 인터페이스라서, 구현체가 무엇이든 사용하는 쪽 코드는 동일하기 때문 —
 * 이것이 이 랩이 보여주려는 핵심(DI + 인터페이스 의존).
 */
@Component
@Profile("lab12")
public class IndexingPipelineDemo implements CommandLineRunner {

    private static final String DOCUMENT_PATH = "classpath:/docs/agentic-rag-survey.pdf";

    // [변경 1] EmbeddingModel 대신 VectorStore 를 주입받는다.
    // spring-ai-starter-vector-store-pgvector 가 classpath 에 있으면
    // Spring 이 PgVectorStore 빈을 자동 구성해서 여기에 꽂아준다.
    // (임베딩 모델은 PgVectorStore 내부에서 알아서 쓰므로 이 클래스가 직접 알 필요가 없다)
    private final VectorStore vectorStore;

    public IndexingPipelineDemo(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    @Override
    public void run(String... args) {
        // 1) 문서 로드
        PagePdfDocumentReader pdfReader = new PagePdfDocumentReader(DOCUMENT_PATH);
        List<Document> documents = pdfReader.get();
        System.out.println("=== 1. 문서 로드 ===");
        System.out.println("로드된 페이지(Document) 수: " + documents.size());
        System.out.println();

        // 2) 청킹
        TokenTextSplitter splitter = TokenTextSplitter.builder()
                .withChunkSize(500)
                .build();
        List<Document> chunks = splitter.apply(documents);
        System.out.println("=== 2. 청킹 결과 ===");
        System.out.println("생성된 청크 수: " + chunks.size());
        for (int i = 0; i < chunks.size(); i++) {
            String preview = chunks.get(i).getText().replaceAll("\\s+", " ");
            preview = preview.substring(0, Math.min(60, preview.length()));
            System.out.printf("  [%d] %s...%n", i, preview);
        }
        System.out.println();

        // 3) VectorStore 저장
        // [변경 2] SimpleVectorStore.builder(embeddingModel).build() 삭제 —
        //         빈을 직접 만들지 않고 주입받은 것을 그대로 쓴다.
        //         add() 호출부는 Day1과 완전히 동일.
        vectorStore.add(chunks);
        System.out.println("=== 3. VectorStore 저장 완료 ===");
        System.out.println("구현체: " + vectorStore.getClass().getSimpleName()
                + " (Day1은 SimpleVectorStore, 지금은 PgVectorStore)");
        System.out.println();

        // 4) 검색 — 이 부분도 Day1과 완전히 동일
        String query = "What is agentic RAG?";
        List<Document> results = vectorStore.similaritySearch(
                SearchRequest.builder().query(query).topK(3).build());

        System.out.println("=== 4. 검색 결과 (질문: \"" + query + "\") ===");
        for (Document doc : results) {
            System.out.println("- " + doc.getText().replaceAll("\\s+", " "));
        }
    }
}