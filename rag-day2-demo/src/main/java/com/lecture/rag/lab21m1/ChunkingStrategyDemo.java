package com.lecture.rag.lab21m1;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * M2.1 — 청킹 전략 4종을 실제 시나리오 문서로 비교.
 * 실행: ./run.sh chunking-strategies
 */
@Component
@Profile("chunking-strategies")
public class ChunkingStrategyDemo implements CommandLineRunner {

    private final EmbeddingModel embeddingModel;

    public ChunkingStrategyDemo(EmbeddingModel embeddingModel) {
        this.embeddingModel = embeddingModel;
    }

    @Override
    public void run(String... args) {
        System.out.println("################ 1. Fixed-size (TokenTextSplitter) vs Recursive ################");
        compareFixedVsRecursive();

        System.out.println();
        System.out.println("################ 2. 문서 구조 활용 청킹 (제N조 경계 / 마크다운 헤더 경계) ################");
        structureBased();

        System.out.println();
        System.out.println("################ 3. Sliding Window (겹치는 청크) ################");
        slidingWindow();

        System.out.println();
        System.out.println("################ 4. Semantic Chunking (문장 간 유사도 급락 지점) ################");
        semanticChunking();
    }

    private Document loadFullText(String file) {
        PagePdfDocumentReader reader = new PagePdfDocumentReader("classpath:/scenarios/" + file);
        List<Document> pages = reader.get();
        String combined = pages.stream().map(Document::getText).reduce("", (a, b) -> a + "\n\n" + b);
        // PDF 텍스트 추출 시 폰트 렌더링 때문에 단어 사이 공백이 비정상적으로 커지는 경우가 있어
        // (예: "물탱크    용량은") 청킹 전에 공백을 정규화하지 않으면 실제 글자 수보다 훨씬 길게 잡힘
        combined = combined.replaceAll("[ \\t]+", " ");
        return new Document(combined);
    }

    /**
     * 마크다운 로더. PDF와 달리 줄바꿈(\n)이 헤더 인식의 기준이므로
     * 공백/탭만 정규화하고 개행은 절대 건드리지 않는다.
     */
    private Document loadMarkdown(String file) {
        try (var in = new ClassPathResource("scenarios/" + file).getInputStream()) {
            String text = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            return new Document(text.replaceAll("[ \\t]+", " "));
        } catch (IOException e) {
            throw new RuntimeException("마크다운 로드 실패: " + file, e);
        }
    }

    private void compareFixedVsRecursive() {
        Document doc = loadFullText("4-terms-startuprecipe.pdf");

        TokenTextSplitter fixed = TokenTextSplitter.builder().withChunkSize(300).build();
        List<Document> fixedChunks = fixed.apply(List.of(doc));
        System.out.println("[Fixed-size] 청크 수: " + fixedChunks.size());
        for (int i = 0; i < Math.min(2, fixedChunks.size()); i++) {
            System.out.println("  [" + i + "] " + preview(fixedChunks.get(i).getText(), 150));
        }

        RecursiveCharacterSplitter recursive = new RecursiveCharacterSplitter(600);
        List<Document> recursiveChunks = recursive.split(doc);
        System.out.println("[Recursive] 청크 수: " + recursiveChunks.size());
        for (int i = 0; i < Math.min(2, recursiveChunks.size()); i++) {
            System.out.println("  [" + i + "] " + preview(recursiveChunks.get(i).getText(), 150));
        }
    }

    /**
     * 같은 StructureBasedSplitter를 Pattern만 바꿔 끼워서 두 가지 구조에 적용.
     * (1) PDF 이용약관 → "제N조" 조항 경계
     * (2) 마크다운 문서 → "#" 헤더 경계
     * 스플리터 코드는 한 줄도 안 바뀐다는 게 이 설계의 핵심.
     */
    private void structureBased() {
        // (1) 제N조 경계
        Document terms = loadFullText("4-terms-startuprecipe.pdf");
        List<Document> articleChunks = StructureBasedSplitter.forKoreanArticles().split(terms);
        System.out.println("[제N조 경계] 청크 수: " + articleChunks.size() + " (실제 조항 수와 비교해볼 것)");
        for (int i = 0; i < Math.min(3, articleChunks.size()); i++) {
            System.out.println("  [" + i + "] " + preview(articleChunks.get(i).getText(), 100));
        }

        // (2) 마크다운 헤더 경계
        System.out.println();
        Document md = loadMarkdown("namu-mangnyeong.md");
        List<Document> headerChunks = StructureBasedSplitter.forMarkdownHeaders().split(md);
        System.out.println("[마크다운 헤더 경계] 청크 수: " + headerChunks.size() + " (실제 헤더 수와 비교해볼 것)");
        for (int i = 0; i < headerChunks.size(); i++) {
            String text = headerChunks.get(i).getText();
            System.out.printf("  [%d] (%4d자) %-22s | %s%n",
                    i, text.length(), headingOf(text), preview(text, 60));
        }

        // 고정 크기와의 차이를 수치로
        System.out.println();
        List<Document> fixedOnMd = TokenTextSplitter.builder().withChunkSize(200).build()
                .apply(List.of(md));
        printStats("Fixed-size(200)", fixedOnMd);
        printStats("헤더 구조", headerChunks);
        System.out.println("  → 구조 청킹은 의미 경계를 지키는 대신 청크 크기가 들쭉날쭉해짐");
    }

    private void slidingWindow() {
        Document doc = loadFullText("6-wiki-jeju.pdf");
        SlidingWindowSplitter splitter = new SlidingWindowSplitter(300, 150);
        List<Document> chunks = splitter.split(doc);
        System.out.println("청크 수: " + chunks.size() + " (window=300자, stride=150자 → 약 50% 겹침)");
        for (int i = 0; i < Math.min(3, chunks.size()); i++) {
            System.out.println("  [" + i + "] " + preview(chunks.get(i).getText(), 120));
        }
        if (chunks.size() >= 2) {
            String end0 = chunks.get(0).getText();
            String start1 = chunks.get(1).getText();
            System.out.println("  --- 청크0 끝부분과 청크1 시작부분이 겹치는지 확인 ---");
            System.out.println("  청크0 끝: ..." + end0.substring(Math.max(0, end0.length() - 60)));
            System.out.println("  청크1 시작: " + start1.substring(0, Math.min(60, start1.length())) + "...");
        }
    }

    private void semanticChunking() {
        Document doc = loadFullText("7-wiki-kimchi.pdf");
        SemanticChunker chunker = new SemanticChunker(embeddingModel, 0.7);
        List<SemanticChunker.SentenceSimilarity> analysis = chunker.analyze(doc);

        System.out.println("문장별 직전 문장과의 코사인 유사도:");
        for (var s : analysis) {
            System.out.printf("  %.4f | %s%n", s.similarityToPrev(), preview(s.sentence(), 60));
        }

        List<Document> chunks = chunker.split(doc);
        System.out.println("threshold=0.7 기준 생성된 청크 수: " + chunks.size());
    }

    /** 청크 첫 줄에서 헤더 텍스트만 뽑기 */
    private String headingOf(String text) {
        String first = text.lines().findFirst().orElse("");
        return first.startsWith("#") ? first.replaceAll("^#+\\s*", "") : "(서문)";
    }

    private void printStats(String label, List<Document> chunks) {
        var stats = chunks.stream().mapToInt(c -> c.getText().length()).summaryStatistics();
        System.out.printf("  %-16s n=%2d  min=%4d  avg=%6.1f  max=%5d%n",
                label, stats.getCount(), stats.getMin(), stats.getAverage(), stats.getMax());
    }

    private String preview(String text, int len) {
        String t = text.replaceAll("\\s+", " ").trim();
        return t.substring(0, Math.min(len, t.length())) + (t.length() > len ? "..." : "");
    }
}