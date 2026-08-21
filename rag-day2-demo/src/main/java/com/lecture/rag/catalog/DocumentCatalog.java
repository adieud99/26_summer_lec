package com.lecture.rag.catalog;

import org.springframework.context.annotation.Profile;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Component;
import java.util.List;

@Component
@Profile({"catalog", "catalog-api"})
public class DocumentCatalog {

    public record DocumentInfo(String documentId, String title, String category,
                               String sourceFile, int chunkCount) {}

    private static final RowMapper<DocumentInfo> ROW_MAPPER = (rs, rowNum) -> new DocumentInfo(
            rs.getString("document_id"),
            rs.getString("title"),
            rs.getString("category"),
            rs.getString("source_file"),
            rs.getInt("chunk_count"));

    private final JdbcTemplate jdbc;

    public DocumentCatalog(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    public void createTableIfAbsent() {
        jdbc.execute("""
                CREATE TABLE IF NOT EXISTS document_catalog (
                    document_id  VARCHAR(64)  PRIMARY KEY,
                    title        VARCHAR(200) NOT NULL,
                    category     VARCHAR(50)  NOT NULL,
                    source_file  VARCHAR(300) NOT NULL,
                    chunk_count  INT          NOT NULL,
                    indexed_at   TIMESTAMP    NOT NULL DEFAULT now()
                )
                """);
    }

    public boolean isRegistered(String documentId) {
        Integer n = jdbc.queryForObject(
                "SELECT count(*) FROM document_catalog WHERE document_id = ?",
                Integer.class, documentId);
        return n != null && n > 0;
    }

    public void register(DocumentInfo info) {
        jdbc.update("""
                INSERT INTO document_catalog (document_id, title, category, source_file, chunk_count)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (document_id) DO UPDATE
                    SET title = EXCLUDED.title,
                        category = EXCLUDED.category,
                        source_file = EXCLUDED.source_file,
                        chunk_count = EXCLUDED.chunk_count
                """, info.documentId(), info.title(), info.category(), info.sourceFile(), info.chunkCount());
    }

    public int countAll() {
        Integer n = jdbc.queryForObject("SELECT count(*) FROM document_catalog", Integer.class);
        return n == null ? 0 : n;
    }

    public List<DocumentInfo> findAll() {
        return jdbc.query("""
                SELECT document_id, title, category, source_file, chunk_count
                FROM document_catalog ORDER BY document_id
                """, ROW_MAPPER);
    }

    public List<DocumentInfo> findByCategory(String category) {
        return jdbc.query("""
                SELECT document_id, title, category, source_file, chunk_count
                FROM document_catalog WHERE category = ? ORDER BY document_id
                """, ROW_MAPPER, category);
    }

    public List<String> findDocumentIdsByCategory(String category) {
        return jdbc.queryForList(
                "SELECT document_id FROM document_catalog WHERE category = ? ORDER BY document_id",
                String.class, category);
    }

    public List<DocumentInfo> searchByTitle(String keyword) {
        return jdbc.query("""
                SELECT document_id, title, category, source_file, chunk_count
                FROM document_catalog WHERE title ILIKE ? ORDER BY document_id
                """, ROW_MAPPER, "%" + keyword + "%");
    }

    public List<String> distinctCategories() {
        return jdbc.queryForList(
                "SELECT DISTINCT category FROM document_catalog ORDER BY category", String.class);
    }

    public DocumentInfo findById(String documentId) {
        List<DocumentInfo> rows = jdbc.query("""
                SELECT document_id, title, category, source_file, chunk_count
                FROM document_catalog WHERE document_id = ?
                """, ROW_MAPPER, documentId);
        return rows.isEmpty() ? null : rows.get(0);
    }

    public List<String> allDocumentIds() {
        return jdbc.queryForList(
                "SELECT document_id FROM document_catalog ORDER BY document_id", String.class);
    }
}