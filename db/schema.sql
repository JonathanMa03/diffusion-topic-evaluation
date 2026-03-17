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
    article_date TEXT,
    article_year INTEGER,
    journal_pub_date TEXT,
    journal_pub_year INTEGER,
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

CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER,
    topic_label TEXT,
    top_terms TEXT,
    n_docs INTEGER,
    model_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(topic_id, model_name)
);

CREATE TABLE IF NOT EXISTS document_topics (
    id INTEGER PRIMARY KEY,
    document_id INTEGER,
    topic_id INTEGER,
    model_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id),
    UNIQUE(document_id, topic_id, model_name)
);