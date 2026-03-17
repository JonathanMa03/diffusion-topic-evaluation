DROP TABLE IF EXISTS document_embeddings;
DROP TABLE IF EXISTS documents;

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source TEXT,
    source_doc_id TEXT,
    title TEXT,
    abstract TEXT,
    publication_year INTEGER,
    publication_date TEXT,
    journal TEXT,
    clean_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, source_doc_id)
);

CREATE TABLE IF NOT EXISTS document_embeddings (
    id INTEGER PRIMARY KEY,
    document_id INTEGER,
    embedding BLOB,
    model_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);