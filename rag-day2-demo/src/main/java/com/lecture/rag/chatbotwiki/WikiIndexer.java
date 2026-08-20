package com.lecture.rag.chatbotwiki;

import org.springframework.ai.document.Document;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

@Component
@Profile("chatbot")
public class WikiIndexer {

    public static final String SOURCE_JEJU = "jeju-wiki";
    public static final String SOURCE_KIMCHI = "kimchi-wiki";

    private static final String JEJU_PATH = "scenarios/6-wiki-jeju.txt";
    private static final String KIMCHI_PATH = "scenarios/7-wiki-kimchi.txt";

    private final VectorStore vectorStore;

    public WikiIndexer(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    public void indexIfAbsent() {
        indexOne(JEJU_PATH, SOURCE_JEJU);
        indexOne(KIMCHI_PATH, SOURCE_KIMCHI);
        System.out.println();
    }

    private void indexOne(String path, String source) {
        if (alreadyIndexed(source)) {
            System.out.println("[인덱싱] " + source + " — 이미 적재되어 있어 건너뜀");
            return;
        }

        Document doc = loadText(path, source);

        List<Document> chunks = TokenTextSplitter.builder()
                .withChunkSize(200)
                .build()
                .apply(List.of(doc));

        vectorStore.add(chunks);
        System.out.println("[인덱싱] " + source + " — 청크 " + chunks.size() + "개 적재 완료");
    }

    private boolean alreadyIndexed(String source) {
        List<Document> hits = vectorStore.similaritySearch(
                SearchRequest.builder()
                        .query(source)
                        .filterExpression("source == '" + source + "'")
                        .topK(1)
                        .similarityThresholdAll()
                        .build());
        return !hits.isEmpty();
    }

    private Document loadText(String path, String source) {
        try (var in = new ClassPathResource(path).getInputStream()) {
            String text = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            return new Document(text, Map.of("source", source));
        } catch (IOException e) {
            throw new RuntimeException("문서 로드 실패: " + path, e);
        }
    }
}