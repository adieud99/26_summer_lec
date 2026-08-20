package com.lecture.rag.catalog;

import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;

@Configuration
@Profile({"catalog", "catalog-api"})
public class CatalogToolsConfig {

    @Bean
    public CatalogTools catalogTools(VectorStore vectorStore, DocumentCatalog catalog) {
        return new CatalogTools(vectorStore, catalog);
    }
}