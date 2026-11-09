from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], repo: Path, helper_dir: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    roots = [str(repo), str(helper_dir)]
    if env.get("PYTHONPATH"):
        roots.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(roots)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Run failed ({result.returncode}); inspect {log_path}\n{' '.join(command)}")


def common_model_args(cfg: dict[str, Any]) -> list[str]:
    return [
        "--dataset", str(cfg["dataset"]), "--data_dir", str(cfg["data_dir"]),
        "--tree_model", str(cfg["tree_model"]), "--child_mode", str(cfg["child_mode"]),
        "--input_features", *map(str, cfg["input_features"]), "--h_size", str(cfg["h_size"]),
        "--image_encoder", str(cfg["image_encoder"]), "--image_size", str(cfg["image_size"]),
        "--embed_dim", str(cfg["embed_dim"]), "--temperature", str(cfg["temperature"]),
        "--sigma_px", str(cfg["sigma_px"]),
    ]


def run_pretraining(cfg, repo, helper_dir, output, seed, test_fold):
    val_fold = (test_fold + 1) % 10
    p = cfg["pretraining"]
    exp = f"ssl_s{seed}_test{test_fold}_val{val_fold}_{time.time_ns()}"
    work_root = Path(cfg["work_dir"]).resolve()
    cmd = [
        cfg["python"], str(helper_dir / "train_cv.py"),
        "--work_dir", str(work_root), "--exp_name", exp,
        "--seed", str(seed), "--gpu", str(cfg["gpu"]),
        "--cv_fold", str(test_fold), "--cv_val_fold", str(val_fold),
        "--batch_size", str(p["batch_size"]), "--epochs", str(p["epochs"]),
        "--lr", str(p["learning_rate"]), "--wd", str(p["weight_decay"]),
        "--warmup_epochs", str(p["warmup_epochs"]),
        "--save_freq", str(p["epochs"]), "--val_freq", str(p["val_freq"]),
        "--use_knn_eval", "--eval_jm", "--knn_k", "5", "--knn_fusion", "concat",
    ] + common_model_args(cfg)
    log = output / "logs" / f"{exp}.log"
    run(cmd, repo, helper_dir, log)

    exp_dir = work_root / exp
    candidates = list(exp_dir.glob("best_JM_epoch_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No validation-selected JM checkpoint found in {exp_dir}; inspect {log}")
    selected = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    dest = output / "checkpoints" / f"ssl_s{seed}_test{test_fold}_val{val_fold}.pth"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected, dest)
    metrics_path = exp_dir / "cv_test_knn.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    return dest, metrics


def run_finetune(cfg, repo, helper_dir, output, seed, test_fold, condition, modality, pi_variant):
    val_fold = (test_fold + 1) % 10
    f = cfg["finetune"]
    if modality == "tree_only":
        mode, fusion = "tree_only", "concat"
    elif modality == "image_only":
        mode, fusion = "image_only", "concat"
    elif modality == "multimodal_concat":
        mode, fusion = "multimodal", "concat"
    elif modality == "multimodal_add":
        mode, fusion = "multimodal", "add"
    else:
        raise ValueError(f"Unknown modality: {modality}")

    exp = f"ft_s{seed}_test{test_fold}_{condition}_{modality}_{pi_variant['name']}_{time.time_ns()}"
    cmd = [
        cfg["python"], str(helper_dir / "finetune_cv.py"),
        "--work_dir", str(Path(cfg["work_dir"]).resolve()), "--exp_name", exp,
        "--dataset", str(cfg["dataset"]), "--data_dir", str(cfg["data_dir"]),
        "--mode", mode, "--fusion_mode", fusion,
        "--seed", str(seed), "--gpu", str(cfg["gpu"]),
        "--cv_fold", str(test_fold), "--cv_val_fold", str(val_fold),
        "--batch_size", str(f["batch_size"]), "--epochs", str(f["epochs"]),
        "--lr", str(f["learning_rate"]), "--wd", str(f["weight_decay"]),
        "--num_workers", str(f["num_workers"]), "--eval_mode", "accuracy",
        "--val_freq", "1", "--early_stopping_patience", str(f["early_stopping_patience"]),
        "--persistence_filters", pi_variant["filters"],
        "--persistence_weight", pi_variant["weights"],
    ]
    if condition == "contrastive":
        checkpoint = output / "checkpoints" / f"ssl_s{seed}_test{test_fold}_val{val_fold}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing pretraining checkpoint: {checkpoint}; run --stage pretrain")
        cmd.extend(["--pretrained_checkpoint", str(checkpoint.resolve())])
    elif condition != "no_pretraining":
        raise ValueError(f"Unknown pretraining condition: {condition}")

    cmd += [
        "--tree_model", str(cfg["tree_model"]), "--child_mode", str(cfg["child_mode"]),
        "--input_features", *map(str, cfg["input_features"]), "--h_size", str(cfg["h_size"]),
        "--image_encoder", str(cfg["image_encoder"]), "--image_size", str(cfg["image_size"]),
        "--embed_dim", str(cfg["embed_dim"]), "--temperature", str(cfg["temperature"]),
        "--sigma_px", str(cfg["sigma_px"]),
    ]
    log = output / "logs" / f"{exp}.log"
    run(cmd, repo, helper_dir, log)
    summary_path = Path(cfg["work_dir"]).resolve() / exp / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    test_metrics = summary.get("test_metrics", {})
    return {
        "seed": seed, "test_fold": test_fold, "val_fold": val_fold,
        "pretraining": condition, "modality": modality,
        "persistence": pi_variant["name"], "filters": pi_variant["filters"],
        "weights": pi_variant["weights"], "val_best_metric": summary.get("val_best_metric"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_macro_f1": test_metrics.get("macro_f1"),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
        "best_epoch": summary.get("best_epoch"), "gpu_peak_mb": summary.get("gpu_peak_mb"),
        "total_seconds": summary.get("total_seconds"), "summary_path": str(summary_path),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_summary(rows, output):
    groups = {}
    keys = ("pretraining", "modality", "persistence")
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for key, group in sorted(groups.items()):
        row = dict(zip(keys, key))
        row["n"] = len(group)
        for metric in ("test_accuracy", "test_macro_f1", "test_balanced_accuracy"):
            vals = [float(item[metric]) for item in group if item.get(metric) is not None]
            row[metric + "_mean"] = statistics.mean(vals) if vals else ""
            row[metric + "_std"] = statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else "")
        summary_rows.append(row)
    write_rows(output / "summary.csv", summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--stage", choices=("pretrain", "finetune", "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--conditions", nargs="+", choices=("contrastive", "no_pretraining"))
    parser.add_argument("--modalities", nargs="+")
    parser.add_argument("--persistence", nargs="+")
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    repo = Path(cfg["repo_root"]).resolve()
    helper_dir = Path(__file__).resolve().parent
    output = Path(cfg["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (repo / "train.py").is_file() or not (repo / "finetune.py").is_file():
        raise FileNotFoundError("repo_root must contain the original train.py and finetune.py")

    seeds = args.seeds or cfg["seeds"]
    folds = args.folds or cfg.get("folds", list(range(10)))
    conditions = args.conditions or cfg["experiments"]["pretraining_conditions"]
    modalities = args.modalities or cfg["experiments"]["modalities"]
    variants = cfg["experiments"]["persistence"]
    if args.persistence:
        wanted = set(args.persistence)
        variants = [v for v in variants if v["name"] in wanted]
        missing = wanted - {v["name"] for v in variants}
        if missing:
            raise ValueError(f"Unknown persistence variants: {sorted(missing)}")

    checkpoints = {}
    pretrain_rows = []
    should_pretrain = args.stage == "pretrain" or (args.stage == "all" and "contrastive" in conditions)
    if should_pretrain:
        for seed in seeds:
            for fold in folds:
                checkpoint, pretrain_metrics = run_pretraining(cfg, repo, helper_dir, output, seed, fold)
                checkpoints[(seed, fold)] = checkpoint
                knn = pretrain_metrics.get("knn_accuracy_percent", {})
                pretrain_rows.append({
                    "seed": seed, "test_fold": fold,
                    "val_fold": (fold + 1) % 10,
                    "best_checkpoint": str(checkpoint),
                    "concat_knn": knn.get("concat"), "add_knn": knn.get("add"),
                    "tree_knn": knn.get("tree_only"), "image_knn": knn.get("image_only"),
                })
                write_rows(output / "pretrain_knn.csv", pretrain_rows)
        if args.stage == "pretrain":
            print(f"Saved fold checkpoints under {output / 'checkpoints'}")
            print(f"Saved frozen-encoder test KNN scores to {output / 'pretrain_knn.csv'}")
            return

    existing_rows = []
    rows_path = output / "runs.csv"
    if args.stage == "finetune" and rows_path.exists():
        with rows_path.open(encoding="utf-8-sig", newline="") as f:
            existing_rows = list(csv.DictReader(f))
    rows = existing_rows
    for seed in seeds:
        for fold in folds:
            if args.stage == "finetune":
                for condition in conditions:
                    if condition == "contrastive" and not (output / "checkpoints" / f"ssl_s{seed}_test{fold}_val{(fold+1)%10}.pth").is_file():
                        raise FileNotFoundError(f"Missing SSL checkpoint for seed={seed}, fold={fold}; run --stage pretrain")
            for condition in conditions:
                for modality in modalities:
                    for variant in variants:
                        rows.append(run_finetune(cfg, repo, helper_dir, output, seed, fold, condition, modality, variant))
                        write_rows(rows_path, rows)
    make_summary(rows, output)
    print(f"Saved fold/seed results to {rows_path}")
    print(f"Saved aggregate results to {output / 'summary.csv'}")


if __name__ == "__main__":
    main()
