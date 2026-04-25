from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from mpi4py import MPI
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, jaccard_score, silhouette_score

try:
    import psutil
except Exception:
    psutil = None


def find_project_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path.cwd().resolve()

    candidates: List[Path] = [start] + list(start.parents)
    for candidate in candidates:
        if (candidate / "Final Datasets").exists() and (candidate / "Milestone 2").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing 'Final Datasets' and 'Milestone 2'.")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or np.isnan(denominator):
        return float("nan")
    return float(numerator / denominator)


def _current_rss_mb() -> float:
    if psutil is None:
        return float("nan")
    return float(psutil.Process().memory_info().rss / (1024.0 * 1024.0))


def load_ground_truth(project_root: Path) -> Dict[str, object]:
    gt_path = project_root / "Final Datasets" / "PATIENT_DOMINANT_CLUSTERS.csv"
    gt_df = pd.read_csv(gt_path)

    hadm_candidates = [c for c in gt_df.columns if "hadm" in c.lower()]
    if not hadm_candidates:
        raise ValueError("HADM column not found in PATIENT_DOMINANT_CLUSTERS.csv")
    hadm_col = hadm_candidates[0]

    if "DOMINANT_CPT_CLUSTER_ID" in gt_df.columns:
        cluster_id_col = "DOMINANT_CPT_CLUSTER_ID"
    else:
        cluster_id_candidates = [c for c in gt_df.columns if "cluster" in c.lower() and c.lower().endswith("id")]
        if not cluster_id_candidates:
            raise ValueError("Cluster ID column not found in PATIENT_DOMINANT_CLUSTERS.csv")
        cluster_id_col = cluster_id_candidates[0]

    cluster_name_candidates = [
        c
        for c in gt_df.columns
        if "cluster" in c.lower() and c != cluster_id_col and gt_df[c].dtype == object
    ]
    cluster_name_col = cluster_name_candidates[0] if cluster_name_candidates else None

    gt_df = gt_df[[hadm_col, cluster_id_col] + ([cluster_name_col] if cluster_name_col else [])].dropna(subset=[cluster_id_col])

    hadm_values = gt_df[hadm_col].astype(np.int64).to_numpy()
    cluster_ids = gt_df[cluster_id_col].astype(np.int64).to_numpy()
    label_lookup = {int(h): int(c) for h, c in zip(hadm_values, cluster_ids)}

    if cluster_name_col:
        id_name_df = (
            gt_df[[cluster_id_col, cluster_name_col]]
            .dropna()
            .drop_duplicates(subset=[cluster_id_col])
            .sort_values(cluster_id_col)
        )
        cluster_id_to_name = {
            int(row[cluster_id_col]): str(row[cluster_name_col]) for _, row in id_name_df.iterrows()
        }
    else:
        cluster_id_to_name = {int(cid): f"Cluster-{int(cid)}" for cid in np.unique(cluster_ids)}

    return {
        "label_lookup": label_lookup,
        "cluster_id_to_name": cluster_id_to_name,
        "n_ground_truth_clusters": int(np.unique(cluster_ids).size),
    }


def load_embedding_dataset(
    project_root: Path,
    embedding: str,
    rows_target: int | None = None,
    lsa_dim: int = 50,
) -> Dict[str, object]:
    embedding = embedding.lower().strip()

    if embedding not in {"bge", "tfidf", "tfidf_lsa50"}:
        raise ValueError("embedding must be one of: bge, tfidf, tfidf_lsa50")

    if embedding == "bge":
        data_path = project_root / "Final Datasets" / "bge_embedding.csv"
    else:
        data_path = project_root / "Final Datasets" / "tfidf_vector.csv"

    read_kwargs: Dict[str, object] = {}
    if rows_target is not None and rows_target > 0:
        read_kwargs["nrows"] = int(rows_target)

    io_start = time.perf_counter()
    df = pd.read_csv(data_path, **read_kwargs)
    io_seconds = time.perf_counter() - io_start

    hadm_candidates = [c for c in df.columns if "hadm" in c.lower()]
    if hadm_candidates:
        hadm_col = hadm_candidates[0]
        hadm_ids = df[hadm_col].astype(np.int64).to_numpy()
        X = df.drop(columns=[hadm_col]).to_numpy(dtype=np.float64)
    else:
        hadm_col = None
        hadm_ids = np.arange(df.shape[0], dtype=np.int64)
        X = df.to_numpy(dtype=np.float64)

    lsa_seconds = 0.0
    lsa_components_used = 0
    if embedding == "tfidf_lsa50":
        lsa_start = time.perf_counter()
        if X.shape[1] <= 2:
            X = X.astype(np.float64)
            lsa_components_used = int(X.shape[1])
        else:
            lsa_components_used = int(min(max(2, lsa_dim), X.shape[1] - 1))
            X = TruncatedSVD(n_components=lsa_components_used, random_state=42).fit_transform(X)
        lsa_seconds = time.perf_counter() - lsa_start

    file_size_mb = float(data_path.stat().st_size / (1024.0 * 1024.0))

    return {
        "X": X,
        "hadm_ids": hadm_ids,
        "rows_used": int(X.shape[0]),
        "feature_count": int(X.shape[1]),
        "source_file": str(data_path),
        "source_file_size_mb": file_size_mb,
        "io_load_seconds": float(io_seconds),
        "io_throughput_mb_s": _safe_div(file_size_mb, io_seconds),
        "embedding": embedding,
        "hadm_column": hadm_col,
        "lsa_seconds": float(lsa_seconds),
        "lsa_components_used": int(lsa_components_used),
    }


def build_partitions(
    n_rows: int,
    world_size: int,
    strategy: str,
    row_weights: np.ndarray | None = None,
) -> List[np.ndarray]:
    strategy = strategy.lower().strip()

    if strategy == "block":
        edges = np.linspace(0, n_rows, world_size + 1, dtype=np.int64)
        return [np.arange(edges[r], edges[r + 1], dtype=np.int64) for r in range(world_size)]

    if strategy == "cyclic":
        return [np.arange(r, n_rows, world_size, dtype=np.int64) for r in range(world_size)]

    if strategy == "load_balanced":
        if row_weights is None or row_weights.shape[0] != n_rows:
            row_weights = np.ones(n_rows, dtype=np.float64)

        order = np.argsort(-row_weights)
        buckets: List[List[int]] = [[] for _ in range(world_size)]
        bucket_loads = np.zeros(world_size, dtype=np.float64)

        for idx in order:
            target = int(np.argmin(bucket_loads))
            buckets[target].append(int(idx))
            bucket_loads[target] += float(row_weights[idx])

        return [np.array(sorted(bucket), dtype=np.int64) for bucket in buckets]

    raise ValueError("distribution strategy must be one of: block, cyclic, load_balanced")


def _compute_row_weights(X: np.ndarray) -> np.ndarray:
    nonzero = np.count_nonzero(np.abs(X) > 1e-12, axis=1).astype(np.float64)
    return np.maximum(nonzero, 1.0)


def logsumexp_stable(a: np.ndarray, axis: int = 1) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True) + 1e-300)
    return np.squeeze(out, axis=axis)


def estimate_log_prob_diag(X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    n_samples = X.shape[0]
    n_components = means.shape[0]
    out = np.empty((n_samples, n_components), dtype=np.float64)
    for k in range(n_components):
        var_k = variances[k]
        mean_k = means[k]
        precision_k = 1.0 / var_k
        log_det = np.sum(np.log(var_k))
        quad = np.sum(((X - mean_k) ** 2) * precision_k, axis=1)
        out[:, k] = -0.5 * (X.shape[1] * np.log(2.0 * np.pi) + log_det + quad)
    return out


def expectation_step_diag(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    log_prob = estimate_log_prob_diag(X, means, variances)
    weighted = log_prob + np.log(weights + 1e-12)
    log_norm = logsumexp_stable(weighted, axis=1)
    resp = np.exp(weighted - log_norm[:, None])
    return resp, log_norm


def predict_diag(X: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    log_prob = estimate_log_prob_diag(X, means, variances)
    weighted = log_prob + np.log(weights + 1e-12)
    return np.argmax(weighted, axis=1).astype(np.int64)


def distributed_diag_gmm_fit_instrumented(
    comm: MPI.Comm,
    X_root: np.ndarray | None,
    n_components: int,
    distribution: str = "block",
    max_iter: int = 120,
    tol: float = 1e-3,
    reg_covar: float = 1e-6,
    random_state: int = 42,
    n_init: int = 1,
) -> Dict[str, object] | None:
    rank = comm.Get_rank()
    size = comm.Get_size()

    total_start = time.perf_counter()

    if rank == 0:
        if X_root is None:
            raise ValueError("X_root must be provided on rank 0")
        X_root = np.asarray(X_root, dtype=np.float64)
        n_rows, n_features = X_root.shape

        row_weights = _compute_row_weights(X_root) if distribution == "load_balanced" else None
        index_partitions = build_partitions(n_rows, size, distribution, row_weights=row_weights)
        x_partitions = [X_root[idx] for idx in index_partitions]
        row_counts = np.array([idx.size for idx in index_partitions], dtype=np.int64)

        seed_rng = np.random.default_rng(random_state)
        init_seeds = seed_rng.integers(1, np.iinfo(np.int32).max, size=max(1, int(n_init)), dtype=np.int64)
    else:
        n_rows = None
        n_features = None
        row_counts = None
        index_partitions = None
        x_partitions = None
        init_seeds = None

    setup_comm_start = time.perf_counter()
    local_idx = comm.scatter(index_partitions, root=0)
    local_x = comm.scatter(x_partitions, root=0)
    n_rows = comm.bcast(n_rows, root=0)
    n_features = comm.bcast(n_features, root=0)
    row_counts = comm.bcast(row_counts, root=0)
    init_seeds = comm.bcast(init_seeds, root=0)
    setup_comm_seconds = time.perf_counter() - setup_comm_start

    local_peak_rss = _current_rss_mb()

    overall_compute_seconds = 0.0
    overall_comm_seconds = setup_comm_seconds
    best_fit: Dict[str, object] | None = None

    for init_run, init_seed in enumerate(init_seeds):
        init_start = time.perf_counter()

        if rank == 0:
            rng = np.random.default_rng(int(init_seed))
            if n_rows >= n_components:
                init_idx = rng.choice(n_rows, size=n_components, replace=False)
            else:
                init_idx = rng.choice(n_rows, size=n_components, replace=True)

            means = X_root[init_idx].copy()
            global_var = np.var(X_root, axis=0) + reg_covar
            variances = np.tile(global_var, (n_components, 1))
            weights = np.full(n_components, 1.0 / n_components, dtype=np.float64)
        else:
            means = None
            variances = None
            weights = None

        init_param_comm_start = time.perf_counter()
        means = comm.bcast(means, root=0)
        variances = comm.bcast(variances, root=0)
        weights = comm.bcast(weights, root=0)
        init_param_comm_seconds = time.perf_counter() - init_param_comm_start

        prev_ll = None
        converged = False
        n_iter = 0

        iter_total_seconds: List[float] = []
        iter_compute_seconds: List[float] = []
        iter_comm_seconds: List[float] = []

        for it in range(max_iter):
            iter_start = time.perf_counter()

            compute_start = time.perf_counter()
            if local_x.shape[0] > 0:
                local_resp, local_log_norm = expectation_step_diag(local_x, weights, means, variances)
                local_nk = local_resp.sum(axis=0)
                local_sum_x = local_resp.T @ local_x
                local_sum_x2 = local_resp.T @ (local_x ** 2)
                local_ll = np.array([local_log_norm.sum()], dtype=np.float64)
            else:
                local_nk = np.zeros(n_components, dtype=np.float64)
                local_sum_x = np.zeros((n_components, n_features), dtype=np.float64)
                local_sum_x2 = np.zeros((n_components, n_features), dtype=np.float64)
                local_ll = np.array([0.0], dtype=np.float64)
            compute_seconds = time.perf_counter() - compute_start

            comm_start = time.perf_counter()
            global_nk = np.zeros_like(local_nk)
            global_sum_x = np.zeros_like(local_sum_x)
            global_sum_x2 = np.zeros_like(local_sum_x2)
            global_ll = np.zeros_like(local_ll)

            comm.Allreduce(local_nk, global_nk, op=MPI.SUM)
            comm.Allreduce(local_sum_x, global_sum_x, op=MPI.SUM)
            comm.Allreduce(local_sum_x2, global_sum_x2, op=MPI.SUM)
            comm.Allreduce(local_ll, global_ll, op=MPI.SUM)
            comm_allreduce_seconds = time.perf_counter() - comm_start

            update_start = time.perf_counter()
            if rank == 0:
                nk_safe = np.maximum(global_nk, 1e-12)
                weights = nk_safe / float(n_rows)
                means = global_sum_x / nk_safe[:, None]
                variances = (global_sum_x2 / nk_safe[:, None]) - (means ** 2)
                variances = np.maximum(variances, reg_covar)

                if prev_ll is not None and abs(global_ll[0] - prev_ll) < tol:
                    converged = True
                prev_ll = float(global_ll[0])
                n_iter = it + 1
            update_seconds = time.perf_counter() - update_start

            bcast_start = time.perf_counter()
            means = comm.bcast(means, root=0)
            variances = comm.bcast(variances, root=0)
            weights = comm.bcast(weights, root=0)
            converged = comm.bcast(converged, root=0)
            n_iter = comm.bcast(n_iter, root=0)
            bcast_seconds = time.perf_counter() - bcast_start

            if rank == 0:
                iter_total_seconds.append(time.perf_counter() - iter_start)
                iter_compute_seconds.append(compute_seconds + update_seconds)
                iter_comm_seconds.append(comm_allreduce_seconds + bcast_seconds)

            local_peak_rss = np.nanmax([local_peak_rss, _current_rss_mb()])

            if converged:
                break

        if local_x.shape[0] > 0:
            local_pred = predict_diag(local_x, weights, means, variances)
        else:
            local_pred = np.empty((0,), dtype=np.int64)

        gather_start = time.perf_counter()
        gathered_pred = comm.gather((local_idx, local_pred), root=0)
        gather_seconds = time.perf_counter() - gather_start

        init_runtime_seconds = time.perf_counter() - init_start

        if rank == 0:
            pred_all = np.full(n_rows, -1, dtype=np.int64)
            for idx_chunk, pred_chunk in gathered_pred:
                pred_all[idx_chunk] = pred_chunk

            if np.any(pred_all < 0):
                raise RuntimeError("Prediction gather integrity check failed.")

            overall_compute_seconds += float(np.sum(iter_compute_seconds))
            overall_comm_seconds += float(init_param_comm_seconds + np.sum(iter_comm_seconds) + gather_seconds)

            current_ll = float("-inf") if prev_ll is None else float(prev_ll)
            if best_fit is None or current_ll > float(best_fit["final_log_likelihood"]):
                best_fit = {
                    "weights": weights.copy(),
                    "means": means.copy(),
                    "variances": variances.copy(),
                    "pred_all": pred_all,
                    "converged": bool(converged),
                    "n_iter": int(n_iter),
                    "final_log_likelihood": current_ll,
                    "best_init_run": int(init_run),
                    "best_init_seed": int(init_seed),
                    "iter_total_seconds": iter_total_seconds,
                    "iter_compute_seconds": iter_compute_seconds,
                    "iter_comm_seconds": iter_comm_seconds,
                    "init_runtime_seconds": float(init_runtime_seconds),
                }

    peak_rss_per_rank = comm.gather(local_peak_rss, root=0)

    if rank == 0:
        if best_fit is None:
            raise RuntimeError("Distributed GMM fitting failed: no valid initialization.")

        total_runtime_seconds = time.perf_counter() - total_start
        total_comm_fraction = _safe_div(overall_comm_seconds, total_runtime_seconds)

        row_counts_arr = np.asarray(row_counts, dtype=np.int64)
        best_fit.update(
            {
                "total_runtime_seconds": float(total_runtime_seconds),
                "total_compute_seconds": float(overall_compute_seconds),
                "total_communication_seconds": float(overall_comm_seconds),
                "communication_fraction": float(total_comm_fraction),
                "setup_comm_seconds": float(setup_comm_seconds),
                "n_init": int(len(init_seeds)),
                "distribution_strategy": distribution,
                "row_counts_per_rank": row_counts_arr.tolist(),
                "rows_per_rank_min": int(row_counts_arr.min()),
                "rows_per_rank_max": int(row_counts_arr.max()),
                "rows_per_rank_mean": float(row_counts_arr.mean()),
                "rows_per_rank_std": float(row_counts_arr.std(ddof=0)),
                "peak_rss_mb_per_rank": [float(x) for x in peak_rss_per_rank],
                "peak_rss_mb_max_rank": float(np.nanmax(peak_rss_per_rank)),
                "peak_rss_mb_mean_rank": float(np.nanmean(peak_rss_per_rank)),
            }
        )
        return best_fit

    return None


def compute_wss(X: np.ndarray, labels: np.ndarray, means: np.ndarray) -> float:
    wss = 0.0
    for k in range(means.shape[0]):
        mask = labels == k
        if mask.sum() == 0:
            continue
        diff = X[mask] - means[k]
        wss += np.sum(diff ** 2)
    return float(wss)


def safe_silhouette(X: np.ndarray, labels: np.ndarray, sample_size: int = 2500) -> float:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2 or X.shape[0] < 3:
        return float("nan")
    actual_sample = min(int(sample_size), int(X.shape[0]))
    return float(silhouette_score(X, labels, sample_size=actual_sample, random_state=42))


def build_dominant_mapping(y_true: np.ndarray, y_pred: np.ndarray, n_clusters: int) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for k in range(n_clusters):
        mask = y_pred == k
        if mask.sum() == 0:
            mapping[k] = -1
            continue
        values, counts = np.unique(y_true[mask], return_counts=True)
        mapping[k] = int(values[np.argmax(counts)])
    return mapping


def remap_predictions(y_pred: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    return np.array([mapping.get(int(p), -1) for p in y_pred], dtype=np.int64)


def cluster_purity(y_true: np.ndarray, y_remapped: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_remapped))


def build_cluster_profile(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    cluster_id_to_name: Dict[int, str],
) -> pd.DataFrame:
    pred_labels = np.asarray(pred_labels)
    gt_labels = np.asarray(gt_labels)

    rows: List[Dict[str, object]] = []
    for pred_cluster in sorted(np.unique(pred_labels).tolist()):
        mask = pred_labels == pred_cluster
        gt_slice = gt_labels[mask]
        if gt_slice.size == 0:
            continue

        values, counts = np.unique(gt_slice, return_counts=True)
        order = np.argsort(-counts)
        values = values[order]
        counts = counts[order]

        p = counts / counts.sum()
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        entropy_norm = float(entropy / np.log(values.size)) if values.size > 1 else 0.0

        top_gt = int(values[0])
        rows.append(
            {
                "pred_cluster": int(pred_cluster),
                "cluster_size": int(mask.sum()),
                "top_gt_cluster_id": top_gt,
                "top_gt_cluster_name": cluster_id_to_name.get(top_gt, f"Cluster-{top_gt}"),
                "top_share": float(counts[0] / counts.sum()),
                "entropy_norm": entropy_norm,
                "distinct_gt_labels": int(values.size),
            }
        )

    return pd.DataFrame(rows).sort_values("pred_cluster").reset_index(drop=True)


def summarize_experiment_metrics(
    fit_out: Dict[str, object],
    X: np.ndarray,
    y_all: np.ndarray,
    cluster_id_to_name: Dict[int, str],
) -> Tuple[Dict[str, object], pd.DataFrame]:
    pred_all = np.asarray(fit_out["pred_all"], dtype=np.int64)
    means = np.asarray(fit_out["means"], dtype=np.float64)

    known_mask = y_all >= 0
    known_rows = int(np.sum(known_mask))
    unmatched_rows = int(y_all.size - known_rows)

    quality_metrics: Dict[str, object] = {
        "wss": compute_wss(X, pred_all, means),
        "silhouette_all": safe_silhouette(X, pred_all),
        "silhouette_known": float("nan"),
        "purity": float("nan"),
        "ari": float("nan"),
        "jaccard_weighted": float("nan"),
        "coherence_top_share_mean": float("nan"),
        "coherence_entropy_mean": float("nan"),
        "known_rows": known_rows,
        "unmatched_rows": unmatched_rows,
    }

    profile_df = pd.DataFrame(
        columns=[
            "pred_cluster",
            "cluster_size",
            "top_gt_cluster_id",
            "top_gt_cluster_name",
            "top_share",
            "entropy_norm",
            "distinct_gt_labels",
        ]
    )

    if known_rows > 0:
        y_known = y_all[known_mask]
        pred_known = pred_all[known_mask]
        quality_metrics["silhouette_known"] = safe_silhouette(X[known_mask], pred_known)

        mapping = build_dominant_mapping(y_known, pred_known, n_clusters=means.shape[0])
        remapped = remap_predictions(pred_known, mapping)

        quality_metrics["purity"] = cluster_purity(y_known, remapped)
        quality_metrics["ari"] = float(adjusted_rand_score(y_known, pred_known))
        quality_metrics["jaccard_weighted"] = float(
            jaccard_score(y_known, remapped, average="weighted", zero_division=0)
        )

        profile_df = build_cluster_profile(pred_known, y_known, cluster_id_to_name)
        if not profile_df.empty:
            quality_metrics["coherence_top_share_mean"] = float(profile_df["top_share"].mean())
            quality_metrics["coherence_entropy_mean"] = float(profile_df["entropy_norm"].mean())

    iter_total = np.asarray(fit_out.get("iter_total_seconds", []), dtype=np.float64)
    iter_compute = np.asarray(fit_out.get("iter_compute_seconds", []), dtype=np.float64)
    iter_comm = np.asarray(fit_out.get("iter_comm_seconds", []), dtype=np.float64)

    iteration_metrics = {
        "avg_iteration_seconds": float(np.nanmean(iter_total)) if iter_total.size else float("nan"),
        "std_iteration_seconds": float(np.nanstd(iter_total)) if iter_total.size else float("nan"),
        "avg_iteration_compute_seconds": float(np.nanmean(iter_compute)) if iter_compute.size else float("nan"),
        "avg_iteration_comm_seconds": float(np.nanmean(iter_comm)) if iter_comm.size else float("nan"),
    }

    merged = {
        "gmm_total_seconds": float(fit_out["total_runtime_seconds"]),
        "compute_seconds": float(fit_out["total_compute_seconds"]),
        "communication_seconds": float(fit_out["total_communication_seconds"]),
        "communication_fraction": float(fit_out["communication_fraction"]),
        "converged": bool(fit_out["converged"]),
        "gmm_iterations": int(fit_out["n_iter"]),
        "log_likelihood": float(fit_out["final_log_likelihood"]),
        "rows_per_rank_min": int(fit_out["rows_per_rank_min"]),
        "rows_per_rank_max": int(fit_out["rows_per_rank_max"]),
        "rows_per_rank_mean": float(fit_out["rows_per_rank_mean"]),
        "rows_per_rank_std": float(fit_out["rows_per_rank_std"]),
        "peak_rss_mb_max_rank": float(fit_out["peak_rss_mb_max_rank"]),
        "peak_rss_mb_mean_rank": float(fit_out["peak_rss_mb_mean_rank"]),
        **iteration_metrics,
        **quality_metrics,
    }

    return merged, profile_df


def append_row_to_csv(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    write_header = not csv_path.exists()
    frame.to_csv(csv_path, mode="a", header=write_header, index=False)
