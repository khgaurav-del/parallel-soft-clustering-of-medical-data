from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")

from mpi4py import MPI

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from m3_utils import (  # noqa: E402
    append_row_to_csv,
    distributed_diag_gmm_fit_instrumented,
    find_project_root,
    load_embedding_dataset,
    load_ground_truth,
    summarize_experiment_metrics,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Milestone 3 MPI experiment runner")
    parser.add_argument("--mode", choices=["strong", "weak", "k"], required=True)
    parser.add_argument("--embedding", choices=["bge", "tfidf", "tfidf_lsa50"], default="bge")
    parser.add_argument("--distribution", choices=["block", "cyclic", "load_balanced"], default="block")
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument("--rows-per-rank", type=int, default=1500)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lsa-dim", type=int, default=50)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--out-file", type=str, required=True)
    parser.add_argument("--profile-out-dir", type=str, default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    project_root = find_project_root(Path.cwd().resolve())
    rows_target = args.rows if args.mode in {"strong", "k"} else args.rows_per_rank * size

    if rank == 0:
        print(
            f"[M3 Runner] mode={args.mode} embedding={args.embedding} dist={args.distribution} "
            f"k={args.n_clusters} ranks={size} rows_target={rows_target}"
        )
        data = load_embedding_dataset(
            project_root=project_root,
            embedding=args.embedding,
            rows_target=rows_target,
            lsa_dim=args.lsa_dim,
        )
        gt = load_ground_truth(project_root)
        y_all = np.array([gt["label_lookup"].get(int(h), -1) for h in data["hadm_ids"]], dtype=np.int64)
        X_root = data["X"]
    else:
        data = None
        gt = None
        y_all = None
        X_root = None

    fit_out = distributed_diag_gmm_fit_instrumented(
        comm=comm,
        X_root=X_root,
        n_components=args.n_clusters,
        distribution=args.distribution,
        max_iter=args.max_iter,
        tol=args.tol,
        reg_covar=args.reg_covar,
        random_state=args.seed,
        n_init=args.n_init,
    )

    if rank == 0:
        summary_metrics, profile_df = summarize_experiment_metrics(
            fit_out=fit_out,
            X=data["X"],
            y_all=y_all,
            cluster_id_to_name=gt["cluster_id_to_name"],
        )

        out_file = Path(args.out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if args.profile_out_dir:
            profile_dir = Path(args.profile_out_dir)
        else:
            profile_dir = out_file.parent / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        profile_path = profile_dir / (
            f"profile_{args.mode}_{args.embedding}_{args.distribution}_p{size}_k{args.n_clusters}_{timestamp}.csv"
        )
        profile_df.to_csv(profile_path, index=False)

        row = {
            "timestamp": timestamp,
            "tag": args.tag,
            "mode": args.mode,
            "embedding": args.embedding,
            "distribution_strategy": args.distribution,
            "mpi_ranks": int(size),
            "n_clusters": int(args.n_clusters),
            "rows_requested": int(rows_target),
            "rows_used": int(data["rows_used"]),
            "feature_count": int(data["feature_count"]),
            "rows_per_rank_target": int(args.rows_per_rank if args.mode == "weak" else -1),
            "n_init": int(args.n_init),
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "reg_covar": float(args.reg_covar),
            "seed": int(args.seed),
            "source_file": data["source_file"],
            "source_file_size_mb": float(data["source_file_size_mb"]),
            "io_load_seconds": float(data["io_load_seconds"]),
            "io_throughput_mb_s": float(data["io_throughput_mb_s"]),
            "lsa_seconds": float(data["lsa_seconds"]),
            "lsa_components_used": int(data["lsa_components_used"]),
            "cluster_profile_file": str(profile_path),
            **summary_metrics,
        }

        append_row_to_csv(out_file, row)

        print(
            f"[M3 Runner] done | total={row['gmm_total_seconds']:.4f}s "
            f"comm={row['communication_seconds']:.4f}s "
            f"iters={row['gmm_iterations']} converged={row['converged']} "
            f"silhouette={row['silhouette_known']:.4f}"
        )
        print(f"[M3 Runner] metrics appended to: {out_file}")
        print(f"[M3 Runner] cluster profile saved to: {profile_path}")


if __name__ == "__main__":
    import numpy as np

    main()
