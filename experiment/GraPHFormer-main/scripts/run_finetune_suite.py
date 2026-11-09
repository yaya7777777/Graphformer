"""
Run a suite of fine-tuning experiments and collect their summaries.

Suites
------
main : step-2 full fine-tuning + the A/B/C modality ablation (step 3 / figure 2)
q4   : concat-fusion classifier vs the dual-encoder contrastive (CLIP) features
q5   : the three persistence filters f1/f2/f3 as the image channel
q7   : batch size / temperature sweep for intra-class compactness

Each config is executed by calling ``finetune.py`` as a subprocess so a single
failure cannot take down the whole suite; per-run metrics land in
``{work_dir}/{exp_name}/summary.json`` and are aggregated into
``{work_dir}/suite_{name}.json`` + a console table.
"""

import argparse
import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNE = os.path.join(_ROOT, "finetune.py")
# the released checkpoints live next to the 'experiment' folder
_PKG = os.path.dirname(os.path.dirname(_ROOT))
CKPT = os.path.join(_PKG, "bil_act_jml_best.pth")

COMMON = dict(
    dataset="JM", ckpt=CKPT, epochs=15, batch_size=8, lr=1e-4,
    temperature=0.07, dropout=0.5, sigma_px=16.0, image_size=518,
)

SUITES = {
    # step 2 (full fine-tuning) + step 3 / figure 2 (A/B/C modality ablation).
    # bs=8 keeps the full-fine-tuning peak at ~9 GB, the ceiling of the 4060 Laptop.
    "main": [
        dict(name="step2_multimodal_concat", mode="multimodal", fusion_mode="concat", epochs=20),
        dict(name="ablation_A_image_only", mode="image_only"),
        dict(name="ablation_B_tree_only", mode="tree_only"),
        dict(name="ablation_C_multimodal_add", mode="multimodal", fusion_mode="add"),
    ],
    # Q4: classifier-on-concatenated-features vs the dual-encoder contrastive features
    "q4": [
        dict(name="q4_concat_classifier", mode="multimodal", fusion_mode="concat", epochs=20),
        dict(name="q4_linear_probe_frozen", mode="multimodal", fusion_mode="concat", epochs=20,
             extra=["--freeze_encoders"]),
        dict(name="q4_concat_frozen_image", mode="multimodal", fusion_mode="concat",
             extra=["--freeze_image_backbone"]),
    ],
}


def build_cmd(cfg, out_root, py):
    c = dict(COMMON)
    c.update({k: v for k, v in cfg.items() if k not in ("name", "extra", "persistence_filters")})
    cmd = [
        py, FINETUNE,
        "--exp_name", cfg["name"],
        "--work_dir", out_root,
        "--dataset", c["dataset"],
        "--pretrained_checkpoint", c["ckpt"],
        "--mode", c["mode"],
        "--epochs", str(c["epochs"]),
        "--batch_size", str(c["batch_size"]),
        "--lr", str(c["lr"]),
        "--temperature", str(c["temperature"]),
        "--dropout", str(c["dropout"]),
        "--image_size", str(c["image_size"]),
        "--sigma_px", str(c["sigma_px"]),
        "--embed_dim", "128",
        "--h_size", "256",
        "--eval_mode", "accuracy",
        "--val_freq", "1",
        "--save_freq", "1000",
        "--num_workers", "0",
        "--seed", "42",
    ]
    if c.get("mode") == "multimodal":
        cmd += ["--fusion_mode", cfg.get("fusion_mode", "concat")]
    if c.get("aug_persistence"):
        cmd += ["--use_persistence_aug"]
    if cfg.get("freeze_image_backbone"):
        cmd += ["--freeze_image_backbone"]
    cmd += cfg.get("extra", [])
    return cmd


def read_summary(out_root, name):
    path = os.path.join(out_root, name, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=list(SUITES))
    ap.add_argument("--work_dir", default="work_dir")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--only", default=None, help="run a single config name")
    args = ap.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    py = sys.executable

    configs = SUITES[args.suite]
    if args.only:
        configs = [c for c in configs if c["name"] == args.only]
    if args.epochs:
        for c in configs:
            c["epochs"] = args.epochs
    if args.batch_size:
        for c in configs:
            c["batch_size"] = args.batch_size

    print(f"suite '{args.suite}': {len(configs)} run(s) -> {work_dir}")
    results = []
    t0 = time.time()
    for i, cfg in enumerate(configs, 1):
        cmd = build_cmd(cfg, work_dir, py)
        print(f"\n[{i}/{len(configs)}] {' '.join(cmd)}", flush=True)
        started = time.time()
        proc = subprocess.run(cmd, cwd=_ROOT)
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        s = read_summary(work_dir, cfg["name"])
        row = dict(name=cfg["name"], mode=cfg.get("mode"), fusion_mode=cfg.get("fusion_mode"),
                   status=status, wall_seconds=time.time() - started)
        if s:
            row.update(best_metric=s.get("best_metric"), best_epoch=s.get("best_epoch"),
                       total_seconds=s.get("total_seconds"), gpu_peak_mb=s.get("gpu_peak_mb"),
                       epoch_seconds_mean=s.get("epoch_seconds_mean"),
                       n_train=s.get("n_train"), n_test=s.get("n_test"),
                       trainable_params=s.get("trainable_params"))
        results.append(row)
        print(f"    -> {status}: {row}", flush=True)

    out = os.path.join(work_dir, f"suite_{args.suite}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(suite=args.suite, configs=configs, results=results,
                       total_seconds=time.time() - t0), f, indent=2, ensure_ascii=False)

    print(f"\n=== suite '{args.suite}' summary ===")
    print(f"{'name':32s} {'status':8s} {'test_acc':>9s} {'epoch':>6s} {'s/epoch':>8s} {'peakMB':>8s}")
    for r in results:
        print(f"{r['name']:32s} {r['status']:8s} "
              f"{(r.get('best_metric') or float('nan')):9.2f} "
              f"{(r.get('best_epoch') or 0):6d} "
              f"{(r.get('epoch_seconds_mean') or float('nan')):8.1f} "
              f"{(r.get('gpu_peak_mb') or float('nan')):8.0f}")
    print("saved ->", out)


if __name__ == "__main__":
    main()
