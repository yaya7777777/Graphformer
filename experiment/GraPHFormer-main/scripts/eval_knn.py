"""
Step 1 - frozen-feature (KNN) evaluation of the released GraPHFormer checkpoints.

Loads each pretrained checkpoint into the CLIP-style model, extracts the frozen
tree / image embeddings and evaluates a k-NN classifier on the Janelia MouseLight
(JM, 4-class) benchmark, in the same three fusion settings the released
checkpoints report in their metadata:

    tree        : tree embedding only
    image       : image (persistence image) embedding only
    concat      : [tree ; image]
    add         : tree + image

Usage:
    python scripts/eval_knn.py --ckpt bil_act_jml_best.pth --out work_dir/eval
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from graphformer.data import LABEL_DICT, NeuronTreeDataset, fold_dirs, get_collate_fn  # noqa: E402
from graphformer.models import CLIPModel  # noqa: E402

#: per-checkpoint config inferred from the released state-dict key shapes
CKPT_CONFIG = {
    "bil_act_jml_best.pth": dict(use_image_projection_head=True),
    "glllia_weights.pth": dict(use_image_projection_head=False),
    "N7_best_weights.pth": dict(use_image_projection_head=False),
}

DEFAULT_ARGS = dict(
    tree_model="double",
    child_mode="sum",
    input_features=[2, 3, 4, 12, 13],
    h_size=256,
    bn=False,
    image_encoder="dinov2_vits14",
    image_size=518,
    freeze_image_backbone=False,
    embed_dim=128,
    temperature=0.07,
    loss_type="clip",
)


class Args(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def build_args(**over):
    a = Args(DEFAULT_ARGS)
    a.update(over)
    return a


def build_model(ckpt_path, args, device):
    model = CLIPModel(args)

    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}

    # the released checkpoints differ in the image projection head layout:
    # a bare nn.Linear, or a one-layer nn.Sequential (-> '.0.' keys)
    if not any(k.startswith("image_projection.2") for k in sd):
        feat_dim = model.image_encoder.encoder.feat_dim
        if any(k.startswith("image_projection.0") for k in sd):
            model.image_projection = torch.nn.Sequential(torch.nn.Linear(feat_dim, args.embed_dim))
        else:
            model.image_projection = torch.nn.Linear(feat_dim, args.embed_dim)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  state_dict: {len(sd)} keys | missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("   MISSING:", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("   UNEXPECTED:", unexpected[:8], "..." if len(unexpected) > 8 else "")

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    meta = {k: v for k, v in ck.items() if k != "state_dict"} if isinstance(ck, dict) else {}
    return model, meta, missing, unexpected


@torch.no_grad()
def extract(model, loader, device):
    tree_f, img_f, labels = [], [], []
    for batch in loader:
        zt = model.tree_projection(model.tree_encoder(batch))
        zi = model.image_projection(model.image_encoder(batch.images))
        tree_f.append(zt.float().cpu())
        img_f.append(zi.float().cpu())
        labels.append(batch.label.cpu())
    return torch.cat(tree_f), torch.cat(img_f), torch.cat(labels)


def knn_scores(zt, zi, y, train_idx, test_idx, k=20):
    def knn(x):
        clf = KNeighborsClassifier(n_neighbors=min(k, len(train_idx)))
        clf.fit(x[train_idx].numpy(), y[train_idx].numpy())
        return clf.score(x[test_idx].numpy(), y[test_idx].numpy()) * 100.0

    def l2(v):
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    ztn, zin = l2(zt), l2(zi)
    return {
        "tree": knn(zt),
        "image": knn(zi),
        "tree_norm": knn(ztn),
        "image_norm": knn(zin),
        "concat": knn(torch.cat([ztn, zin], dim=1)),
        "concat_raw": knn(torch.cat([zt, zi], dim=1)),
        "add": knn(ztn + zin),
        "mul": knn(ztn * zin),
    }


def build_loaders(model, dataset, dirs_train, dirs_test, args_cli, device, reload):
    common = dict(dataset=dataset, label_dict=LABEL_DICT[dataset],
                  image_size=args_cli.image_size, sigma_px=args_cli.sigma_px, reload=reload)
    train_ds = NeuronTreeDataset(phase="train", data_dir={"train": dirs_train}, **common)
    test_ds = NeuronTreeDataset(phase="test", data_dir={"test": dirs_test}, **common)
    collate = get_collate_fn(device)
    tr = DataLoader(train_ds, batch_size=args_cli.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=args_cli.num_workers)
    te = DataLoader(test_ds, batch_size=args_cli.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=args_cli.num_workers)
    return tr, te


def run_split(model, dataset, dirs_train, dirs_test, args_cli, device, reload=False):
    """Feature extraction + KNN for one train/test split."""
    tr_loader, te_loader = build_loaders(model, dataset, dirs_train, dirs_test, args_cli, device, reload)
    t0 = time.time()
    zt_tr, zi_tr, y_tr = extract(model, tr_loader, device)
    zt_te, zi_te, y_te = extract(model, te_loader, device)
    zt = torch.cat([zt_tr, zt_te])
    zi = torch.cat([zi_tr, zi_te])
    y = torch.cat([y_tr, y_te])
    n_tr = len(y_tr)
    scores = knn_scores(zt, zi, y, np.arange(n_tr), np.arange(n_tr, len(y)), args_cli.knn_k)
    return scores, n_tr, int(len(y) - n_tr), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt_dir", default=os.path.dirname(os.path.dirname(_ROOT)))
    ap.add_argument("--dataset", default="JM")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--knn_k", type=int, default=20)
    ap.add_argument("--sigma_px", type=float, default=16.0)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--out", default="work_dir/eval")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--cv", action="store_true",
                    help="10-fold CV (folds 0-7 train / 8-9 test when off)")
    args_cli = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cfg = dict(CKPT_CONFIG.get(os.path.basename(args_cli.ckpt), {}))
    model_args = build_args(image_size=args_cli.image_size, **cfg)
    ckpt_path = args_cli.ckpt
    if not os.path.isabs(ckpt_path) and not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args_cli.ckpt_dir, ckpt_path)

    print(f"checkpoint: {ckpt_path}")
    model, meta, missing, unexpected = build_model(ckpt_path, model_args, device)

    if args_cli.cv:
        per_fold, t0 = [], time.time()
        for fold in range(10):
            tr_dirs = fold_dirs(args_cli.dataset, [f for f in range(10) if f != fold])
            te_dirs = fold_dirs(args_cli.dataset, [fold])
            # reload=True: the on-disk cache is keyed by (dataset, phase), so it
            # must be rebuilt for every fold
            scores, n_tr, n_te, dt = run_split(model, args_cli.dataset, tr_dirs, te_dirs,
                                               args_cli, device, reload=True)
            per_fold.append(scores)
            print(f"  fold {fold}: n_train={n_tr} n_test={n_te} "
                  f"concat={scores['concat']:.2f} add={scores['add']:.2f} tree={scores['tree_norm']:.2f} "
                  f"image={scores['image_norm']:.2f}  ({dt:.0f}s)", flush=True)
        keys = per_fold[0].keys()
        summary = {k: dict(mean=float(np.mean([f[k] for f in per_fold])),
                           std=float(np.std([f[k] for f in per_fold]))) for k in keys}
        print(f"\n=== {os.path.basename(ckpt_path)} | {args_cli.dataset} 10-fold CV ===")
        for k, v in summary.items():
            print(f"  {k:12s} KNN-{args_cli.knn_k}: {v['mean']:6.2f} +/- {v['std']:.2f}")
        print("  released metadata:", {k: meta.get(k) for k in
              ("epoch", "knn_accuracy", "primary_fusion_mode", "concat_accuracy", "add_accuracy")})
        record = dict(checkpoint=os.path.basename(ckpt_path), dataset=args_cli.dataset,
                      protocol="10fold_cv", per_fold=per_fold, summary=summary,
                      knn_k=args_cli.knn_k, image_size=args_cli.image_size,
                      sigma_px=args_cli.sigma_px, released_meta=meta,
                      missing_keys=missing, unexpected_keys=unexpected,
                      total_seconds=time.time() - t0)
        suffix = "cv"
    else:
        tr_dirs = fold_dirs(args_cli.dataset, range(0, 8))
        te_dirs = fold_dirs(args_cli.dataset, range(8, 10))
        scores, n_tr, n_te, dt = run_split(model, args_cli.dataset, tr_dirs, te_dirs,
                                           args_cli, device, reload=True)
        print(f"\n=== {os.path.basename(ckpt_path)} | {args_cli.dataset} "
              f"(train={n_tr}, test={n_te}) ===")
        for k, v in scores.items():
            print(f"  {k:12s} KNN-{args_cli.knn_k}: {v:6.2f}%")
        print(f"  extract time: {dt:.1f}s")
        print("  released metadata:", {k: meta.get(k) for k in
              ("epoch", "knn_accuracy", "primary_fusion_mode", "concat_accuracy", "add_accuracy")})
        record = dict(checkpoint=os.path.basename(ckpt_path), dataset=args_cli.dataset,
                      protocol="2fold", n_train=n_tr, n_test=n_te, scores=scores,
                      knn_k=args_cli.knn_k, image_size=args_cli.image_size,
                      sigma_px=args_cli.sigma_px, released_meta=meta,
                      missing_keys=missing, unexpected_keys=unexpected,
                      extract_seconds=dt)
        suffix = "eval"

    os.makedirs(args_cli.out, exist_ok=True)
    out_path = os.path.join(
        args_cli.out, f"{suffix}_{os.path.basename(ckpt_path).replace('.pth', '')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=str)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
