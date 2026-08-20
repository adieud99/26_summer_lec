package com.lecture.rag.catalog;

import org.springframework.ai.document.Document;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Component
@Profile({"catalog", "catalog-api"})
public class CatalogIndexer {

    private record DocSpec(String documentId, String title, String category, String path) {}

    private static final List<DocSpec> DOCUMENTS = List.of(
            new DocSpec("doc-wiki-jeju",    "제주도",           "위키",   "scenarios/6-wiki-jeju.txt"),
            new DocSpec("doc-wiki-kimchi",  "김치",             "위키",   "scenarios/7-wiki-kimchi.txt"),
            new DocSpec("doc-wiki-nhis",    "국민건강보험",       "위키",   "scenarios/5-wiki-nhis.txt"),
            new DocSpec("doc-terms-recipe", "서비스 이용약관",    "약관",   "scenarios/4-terms-startuprecipe.txt"),
            new DocSpec("doc-manual-ecom",  "이커머스 제품 매뉴얼", "매뉴얼", "scenarios/1-ecommerce-manual.pdf"),
            new DocSpec("doc-paper-eval",   "LLM 에이전트 평가 서베이", "논문", "scenarios/3-research-llm-agent-eval.pdf"),
            new DocSpec("doc-oss-springai", "Spring AI README", "오픈소스", "scenarios/8-opensource-spring-ai-readme.md")
    );

    private final VectorStore vectorStore;
    private final DocumentCatalog catalog;

    public CatalogIndexer(VectorStore vectorStore, DocumentCatalog catalog) {
        this.vectorStore = vectorStore;
        this.catalog = catalog;
    }

    public void indexIfAbsent() {
        catalog.createTableIfAbsent();

        for (DocSpec spec : DOCUMENTS) {
            if (catalog.isRegistered(spec.documentId())) {
                System.out.println("[적재] " + spec.documentId() + " — 이미 등록되어 있어 건너뜀");
                continue;
            }

            List<Document> chunks = loadAndSplit(spec);

            vectorStore.add(chunks);

            catalog.register(new DocumentCatalog.DocumentInfo(
                    spec.documentId(), spec.title(), spec.category(), spec.path(), chunks.size()));

            System.out.println("[적재] " + spec.documentId() + " (" + spec.category() + ") — 청크 "
                    + chunks.size() + "개 + 카탈로그 1행");
        }

        System.out.println("=== 카탈로그 문서 수: " + catalog.countAll()
                + " / 카테고리: " + catalog.distinctCategories() + " ===");
        System.out.println();
    }

    private List<Document> loadAndSplit(DocSpec spec) {
        Map<String, Object> meta = Map.of("document_id", spec.documentId());

        List<Document> source = new ArrayList<>();
        if (spec.path().endsWith(".pdf")) {
            for (Document page : new PagePdfDocumentReader("classpath:/" + spec.path()).get()) {
                Map<String, Object> merged = new HashMap<>(page.getMetadata());
                merged.putAll(meta);
                source.add(new Document(page.getText(), merged));
            }
        } else {
            source.add(new Document(readText(spec.path()), meta));
        }

        return TokenTextSplitter.builder()
                .withChunkSize(300)
                .build()
                .apply(source);
    }

    private String readText(String path) {
        try (var in = new ClassPathResource(path).getInputStream()) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("문서 로드 실패: " + path, e);
        }
    }
}