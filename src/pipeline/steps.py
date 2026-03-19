# src/pipeline/steps.py

from pathlib import Path


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _check_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at: {path}")


def step_ingestion(config):
    print("[STEP] Ingestion")

    import re
    import sqlite3
    import time
    import requests
    import pandas as pd
    import xml.etree.ElementTree as ET

    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    config.data_path.mkdir(parents=True, exist_ok=True)

    config.run_dir.mkdir(parents=True, exist_ok=True)

    run_data_dir = config.run_dir / "data"
    run_metadata_dir = config.run_dir / "metadata"

    run_data_dir.mkdir(parents=True, exist_ok=True)
    run_metadata_dir.mkdir(parents=True, exist_ok=True)

    # --- DB schema bootstrap ---
    conn = sqlite3.connect(config.db_path)
    conn.executescript("""
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
    """)
    conn.commit()

    # --- PubMed config ---
    eutils_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    ncbi_tool = "diffusion_topic_evolution"
    ncbi_email = "REMOVED_EMAIL"   # replace later if desired
    ncbi_api_key = None

    # General PubMed query logic
    if config.pubmed_query.strip():
        pubmed_query = f"({config.pubmed_query}) AND {config.start_year}:{config.end_year}[pdat]"
    else:
        pubmed_query = f"{config.start_year}:{config.end_year}[pdat]"

    query_file = run_metadata_dir / "pubmed_query.txt"
    with open(query_file, "w") as f:
        f.write(pubmed_query)

    print(f"  Saved query to: {query_file}")

    search_page_size = 200
    fetch_batch_size = 25
    target_pmids = 1000
    sleep_seconds = 0.8

    print(f"  Query: {pubmed_query}")
    print(f"  Target PMIDs: {target_pmids}")
    print(f"  Search page size: {search_page_size}")
    print(f"  Fetch batch size: {fetch_batch_size}")

    # --- Helper functions ---
    def safe_text(elem):
        return elem.text.strip() if elem is not None and elem.text is not None else None

    def extract_abstract_text(article):
        abstract_nodes = article.findall(".//Abstract/AbstractText")
        if not abstract_nodes:
            return None

        parts = []
        for node in abstract_nodes:
            label = node.attrib.get("Label")
            text = "".join(node.itertext()).strip()
            if text:
                parts.append(f"{label}: {text}" if label else text)

        return " ".join(parts) if parts else None

    def extract_journal_pub_date(article):
        pub_date = article.find(".//JournalIssue/PubDate")
        if pub_date is None:
            return None, None

        year = safe_text(pub_date.find("Year"))
        month = safe_text(pub_date.find("Month"))
        day = safe_text(pub_date.find("Day"))

        publication_date = "-".join([x for x in [year, month, day] if x])
        publication_year = int(year) if year and year.isdigit() else None

        return publication_date or None, publication_year

    def extract_article_date(article):
        article_date = article.find(".//Article/ArticleDate")
        if article_date is None:
            return None, None

        year = safe_text(article_date.find("Year"))
        month = safe_text(article_date.find("Month"))
        day = safe_text(article_date.find("Day"))

        article_date_str = "-".join([x for x in [year, month, day] if x])
        article_year = int(year) if year and year.isdigit() else None

        return article_date_str or None, article_year

    def parse_pubmed_xml_to_records(xml_text):
        root = ET.fromstring(xml_text)
        articles = root.findall(".//PubmedArticle")

        records = []
        for article in articles:
            pmid = safe_text(article.find(".//PMID"))
            title_node = article.find(".//ArticleTitle")
            title = "".join(title_node.itertext()).strip() if title_node is not None else None
            abstract = extract_abstract_text(article)
            journal = safe_text(article.find(".//Journal/Title"))

            journal_pub_date, journal_pub_year = extract_journal_pub_date(article)
            article_date, article_year = extract_article_date(article)

            publication_year = journal_pub_year if journal_pub_year is not None else article_year
            publication_date = journal_pub_date if journal_pub_date is not None else article_date

            records.append({
                "source": "pubmed",
                "source_doc_id": pmid,
                "title": title,
                "abstract": abstract,
                "publication_year": publication_year,
                "publication_date": publication_date,
                "journal": journal,
                "article_date": article_date,
                "article_year": article_year,
                "journal_pub_date": journal_pub_date,
                "journal_pub_year": journal_pub_year,
            })

        return records

    def fetch_pubmed_batch(pmids, max_retries=4, base_sleep=1.0):
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": ncbi_tool,
            "email": ncbi_email,
        }

        if ncbi_api_key:
            fetch_params["api_key"] = ncbi_api_key

        fetch_url = f"{eutils_base}/efetch.fcgi"

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(fetch_url, params=fetch_params, timeout=60)
                resp.raise_for_status()
                return resp.text
            except requests.exceptions.RequestException as e:
                print(f"   fetch retry {attempt}/{max_retries} failed: {type(e).__name__} - {e}")
                if attempt == max_retries:
                    raise
                time.sleep(base_sleep * attempt)

        raise RuntimeError("fetch_pubmed_batch failed unexpectedly")

    # --- Search PMIDs ---
    all_pmids = []
    retstart = 0

    while len(all_pmids) < target_pmids:
        page_retmax = min(search_page_size, target_pmids - len(all_pmids))

        search_params = {
            "db": "pubmed",
            "term": pubmed_query,
            "retstart": retstart,
            "retmax": page_retmax,
            "retmode": "json",
            "tool": ncbi_tool,
            "email": ncbi_email,
        }

        if ncbi_api_key:
            search_params["api_key"] = ncbi_api_key

        search_resp = requests.get(f"{eutils_base}/esearch.fcgi", params=search_params, timeout=30)
        search_resp.raise_for_status()

        search_data = search_resp.json()
        batch_pmids = search_data["esearchresult"].get("idlist", [])
        total_count = int(search_data["esearchresult"].get("count", "0"))

        if not batch_pmids:
            print("  No more PMIDs returned; stopping early.")
            break

        all_pmids.extend(batch_pmids)
        all_pmids = list(dict.fromkeys(all_pmids))

        print(
            f"  retstart={retstart:4d} | fetched={len(batch_pmids):3d} | "
            f"collected={len(all_pmids):4d} / target={target_pmids} | total_available={total_count}"
        )

        retstart += len(batch_pmids)
        time.sleep(sleep_seconds)

    if not all_pmids:
        conn.close()
        raise ValueError(
            f"No PubMed IDs found for query:\n{pubmed_query}\n\n"
            "Likely causes:\n"
            "- The selected year range returned no matches\n"
            "- PubMed API request failed upstream"
        )

    print(f"  Final PMID count collected: {len(all_pmids)}")

    pmid_file = run_data_dir / "pmids.txt"
    with open(pmid_file, "w") as f:
        for pmid in all_pmids:
            f.write(f"{pmid}\n")

    print(f"  Saved PMID list to: {pmid_file}")
    # --- Fetch XML in batches ---
    all_records = []
    n_batches = (len(all_pmids) + fetch_batch_size - 1) // fetch_batch_size

    for i in range(0, len(all_pmids), fetch_batch_size):
        batch_num = (i // fetch_batch_size) + 1
        pmid_batch = all_pmids[i:i + fetch_batch_size]

        print(f"  Fetching batch {batch_num}/{n_batches} with {len(pmid_batch)} PMIDs...")
        xml_text = fetch_pubmed_batch(pmid_batch)
        batch_records = parse_pubmed_xml_to_records(xml_text)
        all_records.extend(batch_records)

    if not all_records:
        conn.close()
        raise ValueError("PubMed fetch completed, but zero records were parsed from XML.")

    # --- Clean ---
    df_raw = pd.DataFrame(all_records)
    print(f"  Raw parsed records: {len(df_raw)}")
    raw_records_path = run_data_dir / "raw_records.csv"
    df_raw.to_csv(raw_records_path, index=False)
    print(f"  Saved raw parsed records to: {raw_records_path}")

    df = df_raw.copy()
    df = df.dropna(subset=["title", "abstract", "publication_year"])

    for col in [
        "title",
        "abstract",
        "journal",
        "publication_date",
        "article_date",
        "journal_pub_date",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = df[
        (df["title"].str.len() > 0) &
        (df["abstract"].str.len() > 0) &
        (df["title"].str.lower() != "nan") &
        (df["abstract"].str.lower() != "nan")
    ].copy()

    df = df.drop_duplicates(subset=["source", "source_doc_id"])
    df["clean_text"] = df["title"] + " " + df["abstract"]

    if df.empty:
        conn.close()
        raise ValueError(
            "All parsed records were removed during cleaning.\n\n"
            "Likely causes:\n"
            "- abstracts missing for all records\n"
            "- malformed text fields\n"
            "- too-restrictive query/date range"
        )

    print(f"  Cleaned records to insert: {len(df)}")
    cleaned_records_path = run_data_dir / "cleaned_records.csv"
    df.to_csv(cleaned_records_path, index=False)
    print(f"  Saved cleaned records to: {cleaned_records_path}")

    # --- Insert ---
    cols = [
        "source",
        "source_doc_id",
        "title",
        "abstract",
        "publication_year",
        "publication_date",
        "journal",
        "clean_text",
        "article_date",
        "article_year",
        "journal_pub_date",
        "journal_pub_year",
    ]

    df_to_insert = df[cols].copy()

    insert_sql = """
    INSERT OR IGNORE INTO documents (
        source,
        source_doc_id,
        title,
        abstract,
        publication_year,
        publication_date,
        journal,
        clean_text,
        article_date,
        article_year,
        journal_pub_date,
        journal_pub_year
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    before_n = pd.read_sql_query("SELECT COUNT(*) AS n FROM documents", conn)["n"].iloc[0]

    conn.executemany(insert_sql, df_to_insert.values.tolist())
    conn.commit()

    after_n = pd.read_sql_query("SELECT COUNT(*) AS n FROM documents", conn)["n"].iloc[0]

    print(f"  Rows before insert: {before_n}")
    print(f"  Rows attempted: {len(df_to_insert)}")
    print(f"  Rows after insert: {after_n}")
    print(f"  Rows newly added: {after_n - before_n}")

    # --- Validation ---
    year_counts = pd.read_sql_query("""
        SELECT publication_year, COUNT(*) AS n_docs
        FROM documents
        WHERE publication_year BETWEEN ? AND ?
        GROUP BY publication_year
        ORDER BY publication_year
    """, conn, params=[config.start_year, config.end_year])

    dup_df = pd.read_sql_query("""
        SELECT source, source_doc_id, COUNT(*) AS n
        FROM documents
        GROUP BY source, source_doc_id
        HAVING COUNT(*) > 1
        ORDER BY n DESC, source_doc_id
    """, conn)

    print("  Counts by year:")
    for _, row in year_counts.iterrows():
        print(f"   - {int(row['publication_year'])}: {int(row['n_docs'])}")

    if not dup_df.empty:
        conn.close()
        raise ValueError(
            "Duplicate (source, source_doc_id) rows detected after ingestion.\n"
            "This suggests the DB uniqueness constraint or insert logic is inconsistent."
        )

    import json

    summary = {
        "run_name": config.run_name,
        "start_year": config.start_year,
        "end_year": config.end_year,
        "pubmed_query": config.pubmed_query,
        "resolved_pubmed_query": pubmed_query,
        "pmids_collected": len(all_pmids),
        "raw_records": int(len(df_raw)),
        "cleaned_records": int(len(df)),
        "rows_before_insert": int(before_n),
        "rows_after_insert": int(after_n),
        "rows_newly_added": int(after_n - before_n),
        "year_counts": {
            str(int(row["publication_year"])): int(row["n_docs"])
            for _, row in year_counts.iterrows()
        },
    }

    summary_file = run_metadata_dir / "ingestion_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved ingestion summary to: {summary_file}")

    conn.close()
    print("  Ingestion completed successfully with no duplicate document keys")


def step_embeddings(config):
    print("[STEP] Embeddings")

    import pickle
    import sqlite3

    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    _check_file_exists(config.db_path, "Database")

    conn = sqlite3.connect(config.db_path)

    docs_df = pd.read_sql_query("""
        SELECT
            id,
            clean_text,
            publication_year
        FROM documents
        WHERE publication_year BETWEEN ? AND ?
        ORDER BY id
    """, conn, params=[config.start_year, config.end_year])

    if docs_df.empty:
        conn.close()
        raise ValueError(
            f"No documents found in database for years "
            f"{config.start_year}–{config.end_year}.\n\n"
            "Likely causes:\n"
            "- You have not run ingestion yet\n"
            "- The selected date range has no data\n\n"
            "Fix:\n"
            "Run: python -m scripts.run_ingestion"
        )

    existing_df = pd.read_sql_query("""
        SELECT
            document_id
        FROM document_embeddings
        WHERE model_name = ?
    """, conn, params=[config.embedding_model_name])

    existing_ids = set(existing_df["document_id"].astype(int).tolist())

    missing_df = docs_df[~docs_df["id"].isin(existing_ids)].copy()

    print(f"  Total documents in selected range: {len(docs_df)}")
    print(f"  Existing embeddings for model {config.embedding_model_name}: {len(existing_ids)}")
    print(f"  Documents missing embeddings: {len(missing_df)}")

    if missing_df.empty:
        conn.close()
        print("  No new embeddings needed.")
        return

    print(f"  Loading embedding model: {config.embedding_model_name}")
    model = SentenceTransformer(config.embedding_model_name)

    def serialize_embedding(vec: np.ndarray) -> bytes:
        return pickle.dumps(vec.astype(np.float32), protocol=pickle.HIGHEST_PROTOCOL)

    batch_size = 64
    insert_sql = """
    INSERT INTO document_embeddings (
        document_id,
        embedding,
        model_name
    )
    VALUES (?, ?, ?)
    """

    n_total = len(missing_df)
    n_inserted = 0

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_df = missing_df.iloc[start:end]

        texts = batch_df["clean_text"].tolist()
        doc_ids = batch_df["id"].tolist()

        if any(not isinstance(t, str) or not t.strip() for t in texts):
            conn.close()
            raise ValueError(
                f"Found empty or invalid clean_text values in embedding batch "
                f"{start}:{end}."
            )

        batch_embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        rows_to_insert = [
            (int(doc_id), serialize_embedding(emb), config.embedding_model_name)
            for doc_id, emb in zip(doc_ids, batch_embeddings)
        ]

        conn.executemany(insert_sql, rows_to_insert)
        conn.commit()

        n_inserted += len(rows_to_insert)
        print(f"  Inserted batch {start}-{end} | cumulative inserted: {n_inserted}/{n_total}")

    verify_df = pd.read_sql_query("""
        SELECT COUNT(*) AS n
        FROM document_embeddings
        WHERE model_name = ?
    """, conn, params=[config.embedding_model_name])

    conn.close()

    print(f"  Total embeddings stored for model {config.embedding_model_name}: {verify_df['n'].iloc[0]}")


def step_topics(config):
    print("[STEP] Topic Discovery")

    import pickle
    import sqlite3

    import numpy as np
    import pandas as pd
    import hdbscan
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

    _check_file_exists(config.db_path, "Database")

    config.data_path.mkdir(parents=True, exist_ok=True)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    run_data_dir = config.run_dir / "data"
    run_data_dir.mkdir(parents=True, exist_ok=True)

    hdbscan_assignments_path = config.data_path / "hdbscan_assignments.csv"
    hdbscan_lineage_path = config.data_path / "hdbscan_lineage.csv"
    centroids_path = config.data_path / "hdbscan_centroids.pkl"
    lineage_labels_path = config.data_path / "lineage_labels.csv"
    topic_trajectories_path = config.data_path / "topic_trajectories.pkl"

    run_hdbscan_assignments_path = run_data_dir / "hdbscan_assignments.csv"
    run_hdbscan_lineage_path = run_data_dir / "hdbscan_lineage.csv"
    run_centroids_path = run_data_dir / "hdbscan_centroids.pkl"
    run_lineage_labels_path = run_data_dir / "lineage_labels.csv"
    run_topic_trajectories_path = run_data_dir / "topic_trajectories.pkl"

    print(f"  Using DB: {config.db_path}")
    print(f"  Years: {config.start_year} → {config.end_year}")
    print(f"  HDBSCAN min_cluster_size: {config.hdbscan_min_cluster_size}")
    print(f"  HDBSCAN min_samples: {config.hdbscan_min_samples}")

    conn = sqlite3.connect(config.db_path)

    docs_df = pd.read_sql_query("""
        SELECT
            d.id AS document_id,
            d.publication_year,
            d.title,
            d.clean_text,
            e.embedding
        FROM documents d
        JOIN document_embeddings e
            ON d.id = e.document_id
        WHERE e.model_name = ?
          AND d.publication_year BETWEEN ? AND ?
        ORDER BY d.publication_year, d.id
    """, conn, params=[config.embedding_model_name, config.start_year, config.end_year])

    conn.close()

    if docs_df.empty:
        raise ValueError(
            f"No documents with embeddings found for years "
            f"{config.start_year}–{config.end_year} "
            f"and model {config.embedding_model_name}."
        )

    docs_df["embedding"] = docs_df["embedding"].apply(
        lambda x: np.array(pickle.loads(x), dtype=np.float32)
    )

    print(f"  Loaded {len(docs_df)} embedded documents")

    year_counts = docs_df["publication_year"].value_counts().sort_index()
    print("  Documents per year:")
    for year, n in year_counts.items():
        print(f"   - {year}: {n}")

    X_all = np.vstack(docs_df["embedding"].values)
    X_all_norm = normalize(X_all, norm="l2")
    docs_df["embedding_norm"] = list(X_all_norm)

    yearly_clustered = []

    for year in sorted(docs_df["publication_year"].unique()):
        year_df = docs_df[docs_df["publication_year"] == year].copy()
        X_year = np.vstack(year_df["embedding_norm"].values)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.hdbscan_min_cluster_size,
            min_samples=config.hdbscan_min_samples,
            metric="euclidean"
        )

        labels = clusterer.fit_predict(X_year)
        year_df["hdbscan_label"] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())

        print(
            f"  Year {year} | docs={len(year_df)} | "
            f"clusters={n_clusters} | noise_docs={n_noise}"
        )

        yearly_clustered.append(year_df)

    hdbscan_df = pd.concat(yearly_clustered, ignore_index=True)

    assignments_df = hdbscan_df[["document_id", "publication_year", "hdbscan_label"]].copy()
    assignments_df.to_csv(hdbscan_assignments_path, index=False)
    assignments_df.to_csv(run_hdbscan_assignments_path, index=False)

    print(f"  Saved shared HDBSCAN assignments to: {hdbscan_assignments_path}")
    print(f"  Saved run-specific HDBSCAN assignments to: {run_hdbscan_assignments_path}")

    clustered_df = hdbscan_df[hdbscan_df["hdbscan_label"] != -1].copy()

    if clustered_df.empty:
        raise ValueError(
            "All points were labeled as HDBSCAN noise. "
            "Try lowering min_cluster_size or min_samples."
        )

    centroids = []

    for (year, label), group in clustered_df.groupby(["publication_year", "hdbscan_label"]):
        X = np.vstack(group["embedding_norm"].values)
        centroid = X.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        centroids.append({
            "publication_year": int(year),
            "cluster_id": int(label),
            "centroid": centroid.astype(np.float32),
            "n_docs": int(len(group)),
        })

    centroids_df = pd.DataFrame(centroids)

    with open(centroids_path, "wb") as f:
        pickle.dump(centroids_df, f)
    with open(run_centroids_path, "wb") as f:
        pickle.dump(centroids_df, f)

    print(f"  Saved shared centroids to: {centroids_path}")
    print(f"  Saved run-specific centroids to: {run_centroids_path}")
    print(f"  Built {len(centroids_df)} yearly topic centroids")

    years = sorted(centroids_df["publication_year"].unique())
    if len(years) < 1:
        raise ValueError(
            "Need at least two years with non-noise HDBSCAN clusters to build lineage."
        )

    lineage_records = []
    next_lineage_id = 0

    first_year = years[0]
    first_centroids = centroids_df[centroids_df["publication_year"] == first_year].copy()
    lineage_maps = {first_year: {}}

    for _, row in first_centroids.iterrows():
        cid = int(row["cluster_id"])
        lineage_maps[first_year][cid] = next_lineage_id

        lineage_records.append({
            "lineage_id": next_lineage_id,
            "year": int(first_year),
            "cluster_id": cid,
            "n_docs": int(row["n_docs"]),
        })
        next_lineage_id += 1

    print(f"  Initialized {len(first_centroids)} lineages from {first_year}")

    sim_threshold = 0.8

    for idx in range(len(years) - 1):
        y_prev = years[idx]
        y_next = years[idx + 1]

        prev_df = centroids_df[centroids_df["publication_year"] == y_prev].copy()
        next_df = centroids_df[centroids_df["publication_year"] == y_next].copy()

        X_prev = np.vstack(prev_df["centroid"].values)
        X_next = np.vstack(next_df["centroid"].values)

        sim_matrix = cosine_similarity(X_prev, X_next)

        matches = []
        for i, row in enumerate(sim_matrix):
            best_j = np.argmax(row)
            best_sim = float(row[best_j])

            matches.append({
                "cluster_prev": int(prev_df.iloc[i]["cluster_id"]),
                "cluster_next": int(next_df.iloc[best_j]["cluster_id"]),
                "similarity": best_sim,
            })

        matches_df = pd.DataFrame(matches)
        lineage_maps[y_next] = {}

        for _, row in matches_df.iterrows():
            cid_prev = int(row["cluster_prev"])
            cid_next = int(row["cluster_next"])
            sim = float(row["similarity"])

            if sim >= sim_threshold and cid_prev in lineage_maps[y_prev]:
                lineage_maps[y_next][cid_next] = lineage_maps[y_prev][cid_prev]

        all_next_clusters = set(next_df["cluster_id"].astype(int).tolist())
        already_assigned = set(lineage_maps[y_next].keys())
        birth_clusters = sorted(all_next_clusters - already_assigned)

        for cid in birth_clusters:
            lineage_maps[y_next][cid] = next_lineage_id
            next_lineage_id += 1

        for _, row in next_df.iterrows():
            cid = int(row["cluster_id"])
            lineage_id = lineage_maps[y_next][cid]

            lineage_records.append({
                "lineage_id": lineage_id,
                "year": int(y_next),
                "cluster_id": cid,
                "n_docs": int(row["n_docs"]),
            })

        print(
            f"  Matched {y_prev} → {y_next} | "
            f"persistent={sum(matches_df['similarity'] >= sim_threshold)} | "
            f"births={len(birth_clusters)}"
        )

    lineage_df = pd.DataFrame(lineage_records).sort_values(["lineage_id", "year"])

    if lineage_df.empty:
        raise ValueError("Lineage dataframe is empty after topic processing.")

    lineage_df.to_csv(hdbscan_lineage_path, index=False)
    lineage_df.to_csv(run_hdbscan_lineage_path, index=False)

    print(f"  Saved shared lineage to: {hdbscan_lineage_path}")
    print(f"  Saved run-specific lineage to: {run_hdbscan_lineage_path}")
    print(f"  Final lineage rows: {len(lineage_df)}")

    lineage_summary = (
        lineage_df.groupby("lineage_id")
        .agg(
            start_year=("year", "min"),
            end_year=("year", "max"),
            n_years=("year", "nunique"),
            total_docs=("n_docs", "sum"),
        )
        .reset_index()
    )

    custom_stopwords = set(ENGLISH_STOP_WORDS).union({
        "covid", "19", "covid19", "covid-19",
        "sars", "cov", "sars-cov", "sars-cov-2",
        "pandemic", "coronavirus", "disease",
        "study", "studies", "using", "use",
        "background", "objective", "objectives",
        "methods", "results", "conclusion", "conclusions",
        "analysis", "review", "health"
    })

    bad_terms = {
        "covid", "19", "covid 19", "sars", "cov", "sars cov",
        "pandemic", "coronavirus disease"
    }

    def make_lineage_label_from_titles(lineage_id, titles, top_n=4):
        titles = [t for t in titles if isinstance(t, str) and t.strip()]

        if len(titles) == 0:
            return f"Lineage {int(lineage_id)}", ""

        vectorizer = CountVectorizer(
            stop_words=list(custom_stopwords),
            max_features=100,
            ngram_range=(1, 2),
            min_df=1
        )

        X = vectorizer.fit_transform(titles)
        terms = np.array(vectorizer.get_feature_names_out())
        scores = np.asarray(X.sum(axis=0)).ravel()

        ranked_terms = terms[np.argsort(scores)[::-1]].tolist()
        top_terms = [t for t in ranked_terms if t not in bad_terms][:top_n]

        if len(top_terms) == 0:
            return f"Lineage {int(lineage_id)}", ""

        short_name = ", ".join(top_terms[:3])
        full_terms = "; ".join(top_terms)

        return f"Lineage {int(lineage_id)}: {short_name}", full_terms

    doc_lineage_df = assignments_df.merge(
        lineage_df,
        left_on=["publication_year", "hdbscan_label"],
        right_on=["year", "cluster_id"],
        how="left"
    )

    doc_lineage_df = doc_lineage_df.merge(
        docs_df[["document_id", "title", "clean_text"]],
        on="document_id",
        how="left"
    )

    lineage_label_rows = []

    for lineage_id, group in doc_lineage_df.dropna(subset=["lineage_id"]).groupby("lineage_id"):
        lineage_id = int(lineage_id)
        titles = group["title"].dropna().tolist()

        lineage_name, top_terms_str = make_lineage_label_from_titles(
            lineage_id=lineage_id,
            titles=titles,
            top_n=4
        )

        rep_titles = (
            group["title"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(3)
            .tolist()
        )

        summary_row = lineage_summary[lineage_summary["lineage_id"] == lineage_id].iloc[0]

        lineage_label_rows.append({
            "lineage_id": lineage_id,
            "lineage_name": lineage_name,
            "top_terms": top_terms_str,
            "representative_titles": " || ".join(rep_titles),
            "n_docs_total": int(summary_row["total_docs"]),
            "start_year": int(summary_row["start_year"]),
            "end_year": int(summary_row["end_year"]),
            "n_years": int(summary_row["n_years"]),
        })

    lineage_labels_df = pd.DataFrame(lineage_label_rows).sort_values("lineage_id")

    lineage_labels_df.to_csv(lineage_labels_path, index=False)
    lineage_labels_df.to_csv(run_lineage_labels_path, index=False)

    print(f"  Saved shared lineage labels to: {lineage_labels_path}")
    print(f"  Saved run-specific lineage labels to: {run_lineage_labels_path}")

    lineage_name_map = dict(
        zip(lineage_labels_df["lineage_id"].astype(int), lineage_labels_df["lineage_name"])
    )

    topic_trajectories = {}
    centroids_lookup = {
        (int(row["publication_year"]), int(row["cluster_id"])): row["centroid"]
        for _, row in centroids_df.iterrows()
    }

    for lineage_id, group in lineage_df.groupby("lineage_id"):
        group = group.sort_values("year")

        years_seq = []
        trajectory_seq = []

        for _, row in group.iterrows():
            year = int(row["year"])
            cluster_id = int(row["cluster_id"])

            key = (year, cluster_id)
            if key not in centroids_lookup:
                continue

            years_seq.append(year)
            trajectory_seq.append(np.asarray(centroids_lookup[key], dtype=np.float32))

        if len(years_seq) == 0:
            continue

        lineage_id_int = int(lineage_id)
        topic_trajectories[lineage_id_int] = {
            "years": years_seq,
            "trajectory": trajectory_seq,
            "label": lineage_name_map.get(lineage_id_int, f"Lineage {lineage_id_int}"),
        }

    if not topic_trajectories:
        raise ValueError("Topic trajectories could not be built from lineage and centroids.")

    with open(topic_trajectories_path, "wb") as f:
        pickle.dump(topic_trajectories, f)

    with open(run_topic_trajectories_path, "wb") as f:
        pickle.dump(topic_trajectories, f)

    print(f"  Saved shared topic trajectories to: {topic_trajectories_path}")
    print(f"  Saved run-specific topic trajectories to: {run_topic_trajectories_path}")
    print(f"  Built {len(topic_trajectories)} topic trajectories")
    print("  Topic artifacts saved successfully")


def step_diffusion(config):
    print("[STEP] Diffusion Modeling")

    import pickle

    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split

    _check_file_exists(config.data_path, "Data directory")

    traj_path = config.data_path / "topic_trajectories.pkl"
    _check_file_exists(traj_path, "Topic trajectories artifact")

    config.data_path.mkdir(parents=True, exist_ok=True)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    run_data_dir = config.run_dir / "data"
    run_models_dir = config.run_dir / "models"

    run_data_dir.mkdir(parents=True, exist_ok=True)
    run_models_dir.mkdir(parents=True, exist_ok=True)

    future_csv_path = config.data_path / "future_topic_movement.csv"
    future_pkl_path = config.data_path / "future_topic_states.pkl"
    model_path = config.data_path / "diffusion_mlp.pt"

    run_future_csv_path = run_data_dir / "future_topic_movement.csv"
    run_future_pkl_path = run_data_dir / "future_topic_states.pkl"
    run_model_path = run_models_dir / "diffusion_mlp.pt"

    print(f"  Using trajectories: {traj_path}")

    with open(traj_path, "rb") as f:
        topic_trajectories = pickle.load(f)

    if len(topic_trajectories) == 0:
        raise ValueError("Loaded topic trajectories are empty.")

    print(f"  Loaded {len(topic_trajectories)} topic trajectories")

    transition_rows = []

    for topic_id, info in topic_trajectories.items():
        years = info["years"]
        traj = info["trajectory"]
        label = info["label"]

        if len(years) != len(traj):
            raise ValueError(
                f"Topic {topic_id} has mismatched years ({len(years)}) "
                f"and trajectory length ({len(traj)})."
            )

        for t in range(len(years) - 1):
            transition_rows.append({
                "topic_id": topic_id,
                "topic_label": label,
                "year_from": years[t],
                "year_to": years[t + 1],
                "x_t": np.asarray(traj[t], dtype=np.float32),
                "x_t1": np.asarray(traj[t + 1], dtype=np.float32),
            })

    transitions_df = pd.DataFrame(transition_rows)

    if transitions_df.empty:
        raise ValueError("No transition pairs could be constructed from topic trajectories.")

    X_t = np.vstack(transitions_df["x_t"].values).astype(np.float32)
    X_t1 = np.vstack(transitions_df["x_t1"].values).astype(np.float32)

    print(f"  Built {len(transitions_df)} transition pairs")
    print(f"  Transition matrix shape: {X_t.shape}")

    noise_levels = np.array([0.01, 0.03, 0.05, 0.08, 0.12], dtype=np.float32)
    rng = np.random.default_rng(42)

    expanded_rows = []

    for i in range(len(X_t)):
        x_t = X_t[i]
        x_t1 = X_t1[i]
        topic_id = transitions_df.iloc[i]["topic_id"]
        topic_label = transitions_df.iloc[i]["topic_label"]
        year_from = transitions_df.iloc[i]["year_from"]
        year_to = transitions_df.iloc[i]["year_to"]

        for sigma in noise_levels:
            noise = rng.normal(
                loc=0.0,
                scale=float(sigma),
                size=x_t1.shape
            ).astype(np.float32)

            x_t1_noisy = x_t1 + noise

            expanded_rows.append({
                "topic_id": topic_id,
                "topic_label": topic_label,
                "year_from": year_from,
                "year_to": year_to,
                "sigma": float(sigma),
                "x_t": x_t,
                "x_t1_clean": x_t1,
                "x_t1_noisy": x_t1_noisy,
            })

    expanded_df = pd.DataFrame(expanded_rows)

    X_curr = np.vstack(expanded_df["x_t"].values).astype(np.float32)
    X_next_clean = np.vstack(expanded_df["x_t1_clean"].values).astype(np.float32)
    X_next_noisy = np.vstack(expanded_df["x_t1_noisy"].values).astype(np.float32)
    sigmas = expanded_df["sigma"].to_numpy(dtype=np.float32).reshape(-1, 1)

    print(f"  Expanded training set shape: {X_curr.shape}")

    idx = np.arange(len(X_curr))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    class TopicDiffusionDataset(Dataset):
        def __init__(self, X_curr, X_next_noisy, sigmas, X_next_clean):
            self.X_curr = torch.tensor(X_curr, dtype=torch.float32)
            self.X_next_noisy = torch.tensor(X_next_noisy, dtype=torch.float32)
            self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
            self.X_next_clean = torch.tensor(X_next_clean, dtype=torch.float32)

        def __len__(self):
            return len(self.X_curr)

        def __getitem__(self, idx):
            return {
                "x_curr": self.X_curr[idx],
                "x_next_noisy": self.X_next_noisy[idx],
                "sigma": self.sigmas[idx],
                "x_next_clean": self.X_next_clean[idx],
            }

    train_dataset = TopicDiffusionDataset(
        X_curr[train_idx],
        X_next_noisy[train_idx],
        sigmas[train_idx],
        X_next_clean[train_idx],
    )
    val_dataset = TopicDiffusionDataset(
        X_curr[val_idx],
        X_next_noisy[val_idx],
        sigmas[val_idx],
        X_next_clean[val_idx],
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    class DenoiserMLP(nn.Module):
        def __init__(self, dim=384, hidden_dim=512):
            super().__init__()
            input_dim = dim + dim + 1
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim),
            )

        def forward(self, x_curr, x_next_noisy, sigma):
            x = torch.cat([x_curr, x_next_noisy, sigma], dim=1)
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoiserMLP(dim=X_curr.shape[1], hidden_dim=512).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def run_epoch(loader, optimizer=None):
        is_train = optimizer is not None
        model.train() if is_train else model.eval()

        total_loss = 0.0
        total_n = 0

        for batch in loader:
            x_curr = batch["x_curr"].to(device)
            x_next_noisy = batch["x_next_noisy"].to(device)
            sigma = batch["sigma"].to(device)
            x_next_clean = batch["x_next_clean"].to(device)

            with torch.set_grad_enabled(is_train):
                pred = model(x_curr, x_next_noisy, sigma)
                loss = criterion(pred, x_next_clean)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            batch_size = x_curr.size(0)
            total_loss += loss.item() * batch_size
            total_n += batch_size

        return total_loss / total_n

    print("  Training denoiser...")
    epochs = 100
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, optimizer=optimizer)
        val_loss = run_epoch(val_loader, optimizer=None)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch {epoch:3d} | train={train_loss:.6f} | val={val_loss:.6f}")

    model.eval()

    val_preds = []
    val_targets = []
    val_noisy = []

    with torch.no_grad():
        for batch in val_loader:
            x_curr = batch["x_curr"].to(device)
            x_next_noisy = batch["x_next_noisy"].to(device)
            sigma = batch["sigma"].to(device)
            x_next_clean = batch["x_next_clean"].to(device)

            pred = model(x_curr, x_next_noisy, sigma)

            val_preds.append(pred.cpu().numpy())
            val_targets.append(x_next_clean.cpu().numpy())
            val_noisy.append(x_next_noisy.cpu().numpy())

    Y_val_pred = np.vstack(val_preds)
    Y_val_true = np.vstack(val_targets)
    Y_val_noisy = np.vstack(val_noisy)

    mse_noisy = np.mean((Y_val_noisy - Y_val_true) ** 2)
    mse_pred = np.mean((Y_val_pred - Y_val_true) ** 2)

    print(f"  Validation noisy MSE:   {mse_noisy:.8f}")
    print(f"  Validation denoised MSE:{mse_pred:.8f}")

    latest_states = []

    for topic_id, info in topic_trajectories.items():
        years = info["years"]
        traj = info["trajectory"]
        label = info.get("label", f"Lineage {int(topic_id)}")

        # require at least 2 years so the topic has an actual trajectory
        if len(years) < 1:
            continue

        latest_states.append({
            "topic_id": int(topic_id),
            "topic_label": label,
            "latest_year": int(years[-1]),
            "n_years": int(len(years)),
            "x_latest": np.asarray(traj[-1], dtype=np.float32),
        })

    latest_df = pd.DataFrame(latest_states)
    future_df = latest_df.copy()

    if future_df.empty:
        raise ValueError(
            f"No topics found with latest_year == {config.end_year}. "
            "Cannot generate future predictions."
        )

    X_latest = np.vstack(future_df["x_latest"].values).astype(np.float32)

    future_sigma = 0.08
    rng_future = np.random.default_rng(42)

    X_future_noisy = X_latest + rng_future.normal(
        loc=0.0,
        scale=future_sigma,
        size=X_latest.shape
    ).astype(np.float32)

    sigma_future = np.full((len(X_latest), 1), future_sigma, dtype=np.float32)

    with torch.no_grad():
        x_curr_t = torch.tensor(X_latest, dtype=torch.float32, device=device)
        x_next_noisy_t = torch.tensor(X_future_noisy, dtype=torch.float32, device=device)
        sigma_t = torch.tensor(sigma_future, dtype=torch.float32, device=device)

        X_future_pred = model(x_curr_t, x_next_noisy_t, sigma_t).cpu().numpy()

    movement_norm = np.linalg.norm(X_future_pred - X_latest, axis=1)

    def cosine_sim(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return np.nan
        return float(np.dot(a, b) / denom)

    future_results_df = future_df[["topic_id", "topic_label", "latest_year"]].copy()
    future_results_df["future_sigma"] = future_sigma
    future_results_df["movement_norm"] = movement_norm
    future_results_df["cosine_similarity_to_latest"] = [
        cosine_sim(X_future_pred[i], X_latest[i])
        for i in range(len(X_latest))
    ]

    history_df = pd.DataFrame(history)

    model_artifact = {
    "model_state_dict": model.state_dict(),
    "embedding_dim": X_curr.shape[1],
    "epochs": epochs,
    "history": history,
    "validation_noisy_mse": float(mse_noisy),
    "validation_denoised_mse": float(mse_pred),
    }

    print(f"  Saving shared model to: {model_path}")
    torch.save(model_artifact, model_path)

    print(f"  Saving run-specific model to: {run_model_path}")
    torch.save(model_artifact, run_model_path)

    print(f"  Saving shared future movement CSV to: {future_csv_path}")
    future_results_df.to_csv(future_csv_path, index=False)

    print(f"  Saving run-specific future movement CSV to: {run_future_csv_path}")
    future_results_df.to_csv(run_future_csv_path, index=False)

    future_state_artifact = {
        "topic_ids": future_df["topic_id"].tolist(),
        "topic_labels": future_df["topic_label"].tolist(),
        "latest_year": future_df["latest_year"].tolist(),
        "x_latest": X_latest,
        "x_future_noisy": X_future_noisy,
        "x_future_pred": X_future_pred,
        "future_sigma": future_sigma,
        "embedding_dim": X_latest.shape[1],
        "history": history_df,
        "validation_noisy_mse": float(mse_noisy),
        "validation_denoised_mse": float(mse_pred),
    }

    print(f"  Saving shared future topic states to: {future_pkl_path}")
    with open(future_pkl_path, "wb") as f:
        pickle.dump(future_state_artifact, f)

    print(f"  Saving run-specific future topic states to: {run_future_pkl_path}")
    with open(run_future_pkl_path, "wb") as f:
        pickle.dump(future_state_artifact, f)

    print("  Diffusion artifacts saved successfully")
    print(f"   - shared CSV: {future_csv_path}")
    print(f"   - shared PKL: {future_pkl_path}")
    print(f"   - shared model: {model_path}")
    print(f"   - run CSV: {run_future_csv_path}")
    print(f"   - run PKL: {run_future_pkl_path}")
    print(f"   - run model: {run_model_path}")

def step_visualizations(config):
    print("[STEP] Visualizations")

    import pickle
    import textwrap

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import umap

    _check_file_exists(config.data_path, "Data directory")

    traj_path = config.data_path / "topic_trajectories.pkl"
    future_path = config.data_path / "future_topic_movement.csv"
    lineage_path = config.data_path / "hdbscan_lineage.csv"
    lineage_labels_path = config.data_path / "lineage_labels.csv"

    required_inputs = [traj_path, future_path, lineage_path]
    for path in required_inputs:
        _check_file_exists(path, "Visualization input artifact")

    config.outputs_path.mkdir(parents=True, exist_ok=True)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    run_outputs_dir = config.run_dir / "outputs"
    run_outputs_dir.mkdir(parents=True, exist_ok=True)

    print("  Loading artifacts...")
    with open(traj_path, "rb") as f:
        topic_trajectories = pickle.load(f)

    future_results_df = pd.read_csv(future_path)
    lineage_df = pd.read_csv(lineage_path)

    lineage_labels_df = pd.DataFrame()
    if lineage_labels_path.exists():
        lineage_labels_df = pd.read_csv(lineage_labels_path)

    print(f"   - trajectories: {len(topic_trajectories)} topics")
    print(f"   - future movement rows: {len(future_results_df)}")
    print(f"   - lineage rows: {len(lineage_df)}")

    def wrap_label(s, width=24):
        return "\n".join(textwrap.wrap(str(s), width=width))

    # dynamic year markers
    marker_cycle = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]

    movement_lookup = future_results_df.set_index("topic_id")["movement_norm"].to_dict()

    print("  Building UMAP trajectory data...")
    all_points = []
    meta = []

    for topic_id, info in topic_trajectories.items():
        traj = info["trajectory"]
        years = info["years"]
        label = info.get("label", f"Lineage {int(topic_id)}")

        for i, year in enumerate(years):
            all_points.append(traj[i])
            meta.append({
                "topic_id": int(topic_id),
                "label": label,
                "year": int(year),
                "movement_norm": movement_lookup.get(int(topic_id), np.nan),
            })

    if len(all_points) == 0:
        raise ValueError("No trajectory points found for UMAP visualization.")

    X_all = np.vstack(all_points)
    meta_df = pd.DataFrame(meta)

    unique_years = sorted(meta_df["year"].unique())
    year_markers = {
        year: marker_cycle[i % len(marker_cycle)]
        for i, year in enumerate(unique_years)
    }

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(5, len(X_all) - 1),
        min_dist=0.15,
        metric="cosine",
        random_state=42,
    )
    X_2d = reducer.fit_transform(X_all)

    meta_df["x"] = X_2d[:, 0]
    meta_df["y"] = X_2d[:, 1]

    print("  Saving movement plot...")
    plot_df = future_results_df.sort_values("movement_norm", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["topic_label"], plot_df["movement_norm"])
    plt.xlabel("Predicted movement norm")
    plt.ylabel("Topic")
    plt.title("Predicted Topic Movement for the Next Time Step")
    plt.tight_layout()

    shared_path = config.outputs_path / "movement_norm.png"
    run_path = run_outputs_dir / "movement_norm.png"
    plt.savefig(shared_path, dpi=300, bbox_inches="tight")
    plt.savefig(run_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saving cosine similarity plot...")
    plot_df = future_results_df.sort_values("cosine_similarity_to_latest", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["topic_label"], plot_df["cosine_similarity_to_latest"])
    plt.xlabel("Cosine similarity to latest observed period")
    plt.ylabel("Topic")
    plt.title("Predicted Topic Stability Relative to the Latest Observed Period")
    plt.tight_layout()

    shared_path = config.outputs_path / "cosine_sim_latest.png"
    run_path = run_outputs_dir / "cosine_sim_latest.png"
    plt.savefig(shared_path, dpi=300, bbox_inches="tight")
    plt.savefig(run_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saving top UMAP trajectories...")
    top_topic_ids = top_topic_ids = future_results_df["topic_id"].astype(int).tolist()

    top_meta_df = meta_df[meta_df["topic_id"].isin(top_topic_ids)].copy()

    plt.figure(figsize=(13, 9))
    cmap = plt.cm.get_cmap("tab10", max(len(top_topic_ids), 1))

    for i, topic_id in enumerate(top_topic_ids):
        group = top_meta_df[top_meta_df["topic_id"] == topic_id].sort_values("year")
        if group.empty or len(group) < 2:
            continue

        x = group["x"].values
        y = group["y"].values
        movement = group["movement_norm"].iloc[0]
        movement = float(movement) if not pd.isna(movement) else 0.0
        label = group["label"].iloc[0]
        color = cmap(i)

        lw = 2 + 2.5 * movement
        lw = max(lw, 2.5)

        plt.plot(x, y, linewidth=lw, alpha=0.9, color=color)

        # highlight start and end
        plt.scatter(x[0], y[0], s=120, color=color, marker="o", edgecolor="black", linewidth=0.5, zorder=4)
        plt.scatter(x[-1], y[-1], s=140, color=color, marker="X", edgecolor="black", linewidth=0.5, zorder=5)

        for _, row in group.iterrows():
            plt.scatter(
                row["x"],
                row["y"],
                marker=year_markers.get(row["year"], "o"),
                s=100,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=3
            )

        for j in range(len(group) - 1):
            plt.annotate(
                "",
                xy=(x[j + 1], y[j + 1]),
                xytext=(x[j], y[j]),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=lw * 0.6,
                    color=color,
                    alpha=0.8
                )
            )

        last = group.iloc[-1]
        dx = 10 if last["x"] < np.median(top_meta_df["x"]) else -10
        dy = 8 if last["y"] < np.median(top_meta_df["y"]) else -8

        display_label = f"{label} (Δ={movement:.2f})"

        plt.annotate(
            wrap_label(display_label, width=24),
            xy=(last["x"], last["y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec=color,
                alpha=0.95
            )
        )

    for year, marker in year_markers.items():
        plt.scatter([], [], marker=marker, s=90, label=str(year), color="black")

    plt.title("Top Moving Topic Trajectories (UMAP Projection)", fontsize=15)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Year", loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.grid(alpha=0.25)
    plt.tight_layout()

    shared_path = config.outputs_path / "top8_umap.png"
    run_path = run_outputs_dir / "top8_umap.png"
    plt.savefig(shared_path, dpi=300, bbox_inches="tight")
    plt.savefig(run_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saving lineage plot...")
    lineage_summary = (
        lineage_df.groupby("lineage_id")
        .agg(
            start_year=("year", "min"),
            end_year=("year", "max"),
            n_years=("year", "nunique"),
            total_docs=("n_docs", "sum"),
        )
        .reset_index()
    )

    min_year = int(lineage_df["year"].min())
    max_year = int(lineage_df["year"].max())

    def classify_lineage(row):
        if row["n_years"] >= 2:
            return "persistent"
        elif row["start_year"] == min_year and row["end_year"] == min_year:
            return "early_only"
        elif row["start_year"] == max_year and row["end_year"] == max_year:
            return "late_only"
        else:
            return "other"

    lineage_summary["status"] = lineage_summary.apply(classify_lineage, axis=1)

    if (
        not lineage_labels_df.empty
        and {"lineage_id", "lineage_name"}.issubset(lineage_labels_df.columns)
    ):
        lineage_name_df = lineage_labels_df[["lineage_id", "lineage_name"]].drop_duplicates().copy()
    else:
        lineage_name_df = pd.DataFrame({
            "lineage_id": sorted(lineage_df["lineage_id"].unique())
        })
        lineage_name_df["lineage_name"] = lineage_name_df["lineage_id"].apply(
            lambda x: f"Lineage {int(x)}"
        )

    lineage_plot_df = lineage_df.merge(
        lineage_summary[["lineage_id", "status", "total_docs"]],
        on="lineage_id",
        how="left",
    ).merge(
        lineage_name_df,
        on="lineage_id",
        how="left",
    )

    lineage_plot_df["lineage_name"] = lineage_plot_df["lineage_name"].fillna(
        lineage_plot_df["lineage_id"].apply(lambda x: f"Lineage {int(x)}")
    )

    status_colors = {
        "persistent": "tab:blue",
        "early_only": "tab:red",
        "late_only": "tab:green",
        "other": "gray",
    }

    plt.figure(figsize=(14, 7))

    for lineage_id, group in lineage_plot_df.groupby("lineage_id"):
        group = group.sort_values("year")

        status = group["status"].iloc[0]
        color = status_colors.get(status, "gray")

        lineage_name = group["lineage_name"].iloc[0]
        total_docs = int(group["total_docs"].iloc[0])

        x = group["year"].values
        y = np.full(len(group), lineage_id)

        line_width = 3 if status == "persistent" else 2

        if len(group) > 1:
            plt.plot(x, y, color=color, linewidth=line_width, alpha=0.9)

        plt.scatter(
            x,
            y,
            s=(np.sqrt(group["n_docs"].values) * 20) + 20,
            color=color,
            alpha=0.95,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        for _, row in group.iterrows():
            plt.annotate(
                f'n={int(row["n_docs"])}',
                xy=(row["year"], lineage_id),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )

        plt.annotate(
            f"{lineage_name} (total n={total_docs})",
            xy=(min_year - 0.25, lineage_id),
            xytext=(-5, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec=color,
                alpha=0.92,
            ),
        )

    plt.xlim(min_year - 0.45, max_year + 0.25)
    plt.xticks(sorted(lineage_df["year"].unique()))
    plt.xlabel("Year")
    plt.ylabel("Lineage ID")
    plt.title("Dynamic Topic Lineages (HDBSCAN + Semantic Labels)", fontsize=14)
    plt.grid(axis="x", alpha=0.2)
    plt.subplots_adjust(left=0.40)

    shared_path = config.outputs_path / "linneage.png"
    run_path = run_outputs_dir / "linneage.png"
    plt.savefig(shared_path, dpi=300, bbox_inches="tight")
    plt.savefig(run_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saved shared outputs:")
    print(f"   - {config.outputs_path / 'movement_norm.png'}")
    print(f"   - {config.outputs_path / 'cosine_sim_latest.png'}")
    print(f"   - {config.outputs_path / 'top8_umap.png'}")
    print(f"   - {config.outputs_path / 'linneage.png'}")

    print("  Saved run-specific outputs:")
    print(f"   - {run_outputs_dir / 'movement_norm.png'}")
    print(f"   - {run_outputs_dir / 'cosine_sim_latest.png'}")
    print(f"   - {run_outputs_dir / 'top8_umap.png'}")
    print(f"   - {run_outputs_dir / 'linneage.png'}")