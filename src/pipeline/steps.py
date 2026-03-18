# src/pipeline/steps.py

from pathlib import Path


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _check_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at: {path}")


def step_ingestion(config):
    print("[STEP] Ingestion")
    print(f"  Target years: {config.start_year} → {config.end_year}")
    print("  Status: not implemented yet")

    raise NotImplementedError(
        "Ingestion step is not implemented yet. "
        "Next action: move notebook 01 data ingestion logic into step_ingestion(config)."
    )


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
            f"{config.start_year}–{config.end_year}."
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

    _check_file_exists(config.db_path, "Database")

    hdbscan_assignments_path = config.data_path / "hdbscan_assignments.csv"
    hdbscan_lineage_path = config.data_path / "hdbscan_lineage.csv"

    config.data_path.mkdir(parents=True, exist_ok=True)

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

    # normalize embeddings
    X_all = np.vstack(docs_df["embedding"].values)
    X_all_norm = normalize(X_all, norm="l2")
    docs_df["embedding_norm"] = list(X_all_norm)

    # cluster each year independently
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

    # save assignments
    hdbscan_df[["document_id", "publication_year", "hdbscan_label"]].to_csv(
        hdbscan_assignments_path,
        index=False
    )
    print(f"  Saved HDBSCAN assignments to: {hdbscan_assignments_path}")

    # filter out noise points
    clustered_df = hdbscan_df[hdbscan_df["hdbscan_label"] != -1].copy()

    if clustered_df.empty:
        raise ValueError(
            "All points were labeled as HDBSCAN noise. "
            "Try lowering min_cluster_size or min_samples."
        )

    # compute centroids per (year, cluster)
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

    print(f"  Built {len(centroids_df)} yearly topic centroids")

    years = sorted(centroids_df["publication_year"].unique())
    if len(years) < 2:
        raise ValueError(
            "Need at least two years with non-noise HDBSCAN clusters to build lineage."
        )

    # initialize lineage from first year
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

    # match adjacent years
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

        # propagate persistent matches
        for _, row in matches_df.iterrows():
            cid_prev = int(row["cluster_prev"])
            cid_next = int(row["cluster_next"])
            sim = float(row["similarity"])

            if sim >= sim_threshold and cid_prev in lineage_maps[y_prev]:
                lineage_maps[y_next][cid_next] = lineage_maps[y_prev][cid_prev]

        # births for unmatched next-year clusters
        all_next_clusters = set(next_df["cluster_id"].astype(int).tolist())
        already_assigned = set(lineage_maps[y_next].keys())
        birth_clusters = sorted(all_next_clusters - already_assigned)

        for cid in birth_clusters:
            lineage_maps[y_next][cid] = next_lineage_id
            next_lineage_id += 1

        # save clean records for y_next
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

    lineage_df.to_csv(hdbscan_lineage_path, index=False)
    print(f"  Saved lineage to: {hdbscan_lineage_path}")

    # lightweight validation
    if lineage_df.empty:
        raise ValueError("Lineage dataframe is empty after topic processing.")

    print(f"  Final lineage rows: {len(lineage_df)}")


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

    future_csv_path = config.data_path / "future_topic_movement.csv"
    future_pkl_path = config.data_path / "future_topic_states.pkl"
    model_path = config.data_path / "diffusion_mlp.pt"

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

        if len(years) == 0:
            continue

        latest_states.append({
            "topic_id": topic_id,
            "topic_label": info["label"],
            "latest_year": years[-1],
            "x_latest": np.asarray(traj[-1], dtype=np.float32),
        })

    latest_df = pd.DataFrame(latest_states)
    future_df = latest_df[latest_df["latest_year"] == config.end_year].copy()

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

    print(f"  Saving model to: {model_path}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": X_curr.shape[1],
            "epochs": epochs,
            "history": history,
            "validation_noisy_mse": float(mse_noisy),
            "validation_denoised_mse": float(mse_pred),
        },
        model_path,
    )

    print(f"  Saving future movement CSV to: {future_csv_path}")
    future_results_df.to_csv(future_csv_path, index=False)

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

    print(f"  Saving future topic states to: {future_pkl_path}")
    with open(future_pkl_path, "wb") as f:
        pickle.dump(future_state_artifact, f)

    print("  Diffusion artifacts saved successfully")


def step_visualizations(config):
    print("[STEP] Visualizations")

    import pickle
    import textwrap

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    _check_file_exists(config.data_path, "Data directory")

    traj_path = config.data_path / "topic_trajectories.pkl"
    future_path = config.data_path / "future_topic_movement.csv"
    lineage_path = config.data_path / "hdbscan_lineage.csv"

    required_inputs = [traj_path, future_path, lineage_path]
    for path in required_inputs:
        _check_file_exists(path, "Visualization input artifact")

    config.outputs_path.mkdir(parents=True, exist_ok=True)

    print("  Loading artifacts...")
    with open(traj_path, "rb") as f:
        topic_trajectories = pickle.load(f)

    future_results_df = pd.read_csv(future_path)
    lineage_df = pd.read_csv(lineage_path)

    print(f"   - trajectories: {len(topic_trajectories)} topics")
    print(f"   - future movement rows: {len(future_results_df)}")
    print(f"   - lineage rows: {len(lineage_df)}")

    def wrap_label(s, width=24):
        return "\n".join(textwrap.wrap(s, width=width))

    year_markers = {
        2020: "o",
        2021: "s",
        2022: "^",
    }

    movement_lookup = future_results_df.set_index("topic_id")["movement_norm"].to_dict()

    print("  Building PCA trajectory data...")
    all_points = []
    meta = []

    for topic_id, info in topic_trajectories.items():
        traj = info["trajectory"]
        years = info["years"]
        label = info["label"]

        for i, year in enumerate(years):
            all_points.append(traj[i])
            meta.append({
                "topic_id": topic_id,
                "label": label,
                "year": year,
                "movement_norm": movement_lookup.get(topic_id, np.nan),
            })

    if len(all_points) == 0:
        raise ValueError("No trajectory points found for PCA visualization.")

    X_all = np.vstack(all_points)
    meta_df = pd.DataFrame(meta)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_all)

    meta_df["x"] = X_2d[:, 0]
    meta_df["y"] = X_2d[:, 1]

    print("  Saving movement plot...")
    plot_df = future_results_df.sort_values("movement_norm", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["topic_label"], plot_df["movement_norm"])
    plt.xlabel("Predicted movement norm")
    plt.ylabel("Topic")
    plt.title("Predicted 2023 Topic Movement (Neural Denoiser)")
    plt.tight_layout()
    plt.savefig(config.outputs_path / "movement_norm.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saving cosine similarity plot...")
    plot_df = future_results_df.sort_values("cosine_similarity_to_latest", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["topic_label"], plot_df["cosine_similarity_to_latest"])
    plt.xlabel("Cosine similarity to 2022 state")
    plt.ylabel("Topic")
    plt.title("Predicted 2023 Stability Relative to 2022")
    plt.tight_layout()
    plt.savefig(config.outputs_path / "cosine_sim_latest.png", dpi=300, bbox_inches="tight")
    plt.close()    

    print("  Saving top-8 PCA trajectories...")
    TOP_K = 8
    top_topic_ids = (
        future_results_df
        .sort_values("movement_norm", ascending=False)
        .head(TOP_K)["topic_id"]
        .tolist()
    )

    top_meta_df = meta_df[meta_df["topic_id"].isin(top_topic_ids)].copy()

    plt.figure(figsize=(13, 9))
    cmap = plt.cm.get_cmap("tab10", len(top_topic_ids))

    for i, topic_id in enumerate(top_topic_ids):
        group = top_meta_df[top_meta_df["topic_id"] == topic_id].sort_values("year")

        x = group["x"].values
        y = group["y"].values
        label = group["label"].iloc[0]
        movement = group["movement_norm"].iloc[0]
        color = cmap(i)

        lw = 2 + 2.5 * movement

        plt.plot(x, y, linewidth=lw, alpha=0.9, color=color)

        for _, row in group.iterrows():
            plt.scatter(
                row["x"], row["y"],
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
        dx = 10 if last["x"] < 0.3 else -10
        dy = 8 if last["y"] < 0 else -8

        plt.annotate(
            wrap_label(label, width=22),
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

    plt.title("Top 8 Moving Topic Trajectories (PCA Projection)", fontsize=15)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Year", loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(config.outputs_path / "top8_pca.png", dpi=300, bbox_inches="tight")
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

    def classify_lineage(row):
        if row["n_years"] >= 2:
            return "persistent"
        elif row["start_year"] == 2020 and row["end_year"] == 2020:
            return "dies_after_2020"
        elif row["start_year"] == 2021 and row["end_year"] == 2021:
            return "born_in_2021"
        else:
            return "other"

    lineage_summary["status"] = lineage_summary.apply(classify_lineage, axis=1)

    lineage_name_map = {
        0: "Lineage 0: Medical imaging cluster",
        1: "Lineage 1: Mobility and network spread cluster",
        2: "Lineage 2: Cross-sectional public health cluster",
        3: "Lineage 3: Main clinical care cluster",
        4: "Lineage 4: Digital media and infodemic cluster",
        5: "Lineage 5: Online biomedical education cluster",
        6: "Lineage 6: Financial markets cluster",
    }

    lineage_name_df = pd.DataFrame(
        [{"lineage_id": k, "lineage_name": v} for k, v in lineage_name_map.items()]
    )

    lineage_plot_df = lineage_df.merge(
        lineage_summary[["lineage_id", "status", "total_docs"]],
        on="lineage_id",
        how="left"
    ).merge(
        lineage_name_df,
        on="lineage_id",
        how="left"
    )

    lineage_plot_df["lineage_name"] = lineage_plot_df["lineage_name"].fillna(
        lineage_plot_df["lineage_id"].apply(lambda x: f"Lineage {int(x)}")
    )

    status_colors = {
        "persistent": "tab:blue",
        "dies_after_2020": "tab:red",
        "born_in_2021": "tab:green",
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
            s=group["n_docs"].values * 7,
            color=color,
            alpha=0.95,
            edgecolor="black",
            linewidth=0.5,
            zorder=3
        )

        for _, row in group.iterrows():
            plt.annotate(
                f'n={int(row["n_docs"])}',
                xy=(row["year"], lineage_id),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color=color
            )

        plt.annotate(
            f"{lineage_name} (total n={total_docs})",
            xy=(2019.74, lineage_id),
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
                alpha=0.92
            )
        )

    plt.xlim(2019.55, 2021.08)
    plt.xticks([2020, 2021])
    plt.xlabel("Year")
    plt.ylabel("Lineage ID")
    plt.title("Dynamic Topic Lineages (HDBSCAN + Semantic Labels)", fontsize=14)
    plt.grid(axis="x", alpha=0.2)
    plt.subplots_adjust(left=0.38)
    plt.savefig(config.outputs_path / "linneage.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saved outputs:")
    print(f"   - {config.outputs_path / 'movement_norm.png'}")
    print(f"   - {config.outputs_path / 'cosine_sim_latest.png'}")
    print(f"   - {config.outputs_path / 'top8_pca.png'}")
    print(f"   - {config.outputs_path / 'linneage.png'}")