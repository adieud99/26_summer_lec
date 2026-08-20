package com.lecture.rag.lab21;

import com.lecture.rag.lab21m1.StructureBasedSplitter;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Lab 2.1 — Day1의 SimpleVectorStore를 PgVectorStore로 교체.
 * VectorStore는 인터페이스라서 add()/similaritySearch() 호출부는 Day1과 완전히 동일 —
 * 바뀐 건 Bean 구현체(PgVectorStore가 자동 주입됨)뿐이라는 게 이 랩의 핵심 포인트.
 *
 * 실행: 1) docker compose up -d  (PGVector 컨테이너 기동)
 *       2) ./run.sh lab21
 *
 * 영속성 비교 실습: 이 프로필로 한 번 인덱싱한 뒤 앱을 완전히 종료했다가 다시 실행해보면,
 * "재인덱싱하지 않았는데도" 검색이 바로 되는 걸 확인할 수 있다 (SimpleVectorStore였다면
 * 재시작 시 데이터가 전부 사라져서 매번 재인덱싱해야 했음).
 *
 * Lab2.0 시나리오: namu-mangnyeong.md (마크다운) — PDF + 고정 크기 대신
 * 마크다운 헤더 경계로 구조 청킹해서 적재한다.
 */
@Component
@Profile("lab21")
public class PgVectorIndexingDemo implements CommandLineRunner {

    /** Lab2.0에서 고른 내 시나리오 문서 */
    private static final String SCENARIO = "scenarios/namu-mangnyeong.md";

    /** 이 문서에 실제로 등장하는 단어여야 countExisting()이 제대로 동작함 */
    private static final String PROBE_KEYWORD = "망나뇽";

    private final VectorStore vectorStore; // PgVectorStore가 자동 주입됨 (application.yml 설정 기반)

    public PgVectorIndexingDemo(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    @Override
    public void run(String... args) {
        long existingCount = countExisting();
        System.out.println("=== 시작 시점 PGVector에 이미 저장된 문서 수: " + existingCount + " ===");

        if (existingCount == 0) {
            System.out.println("=== 처음 실행이라 인덱싱을 진행합니다 ===");
            index();
        } else {
            System.out.println("=== 이미 인덱싱된 데이터가 있습니다 — 재인덱싱 없이 바로 검색합니다 ===");
            System.out.println("(SimpleVectorStore였다면 재시작 시 이 데이터가 전부 사라졌을 것 — 이게 영속성의 차이)");
        }
        System.out.println();

        // 마크다운 구조 청킹이라 각 청크가 곧 하나의 섹션 — 질문마다 다른 섹션이 걸려야 정상
        String[] queries = {
                "망나뇽의 타입이 뭐야?",
                "멀티스케일 특성이 왜 강한 거야?",
                "미뇽이랑 생긴 게 왜 이렇게 달라?"
        };

        for (String query : queries) {
            List<Document> results = vectorStore.similaritySearch(
                    SearchRequest.builder().query(query).topK(2).build());

            System.out.println("=== 검색 결과 (질문: \"" + query + "\") ===");
            for (Document doc : results) {
                System.out.println("- " + preview(doc.getText(), 120));
            }
            System.out.println();
        }
    }

    private void index() {
        Document doc = loadMarkdown(SCENARIO);

        // PDF + 고정 크기(TokenTextSplitter) 대신, 마크다운은 헤더 경계로 자른다.
        // 섹션 하나가 곧 청크 하나가 되므로 검색 결과가 의미 단위로 떨어진다.
        List<Document> chunks = StructureBasedSplitter.forMarkdownHeaders().split(doc);

        vectorStore.add(chunks);
        System.out.println("생성된 청크 수: " + chunks.size());
        for (int i = 0; i < chunks.size(); i++) {
            System.out.printf("  [%d] %s%n", i, preview(chunks.get(i).getText(), 60));
        }
    }

    /**
     * 마크다운 로더. PDF와 달리 줄바꿈(\n)이 헤더 인식의 기준이므로
     * 공백/탭만 정규화하고 개행은 건드리지 않는다.
     */
    private Document loadMarkdown(String path) {
        try (var in = new ClassPathResource(path).getInputStream()) {
            String text = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            return new Document(text.replaceAll("[ \\t]+", " "));
        } catch (IOException e) {
            throw new RuntimeException("문서 로드 실패: " + path, e);
        }
    }

    private long countExisting() {
        // 빈 질의로 top-1000 검색해서 기존 저장된 문서 수를 어림잡음 (PgVectorStore엔 count() API가 없음)
        List<Document> all = vectorStore.similaritySearch(
                SearchRequest.builder().query(PROBE_KEYWORD).topK(1000).similarityThresholdAll().build());
        return all.size();
    }

    private String preview(String text, int len) {
        String t = text.replaceAll("\\s+", " ").trim();
        return t.substring(0, Math.min(len, t.length())) + (t.length() > len ? "..." : "");
    }
}