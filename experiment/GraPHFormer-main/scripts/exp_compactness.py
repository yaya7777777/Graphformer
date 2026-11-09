"""
Q7 (figure 5) - how batch size and temperature shape embedding compactness.

For each (batch_size, temperature) pair this trains the CLIP-style model with the
contrastive objective for a few epochs (starting from the released checkpoint, so
convergence is fast and the batch/temperature effect is isolated), then measures
class compactness of the tree / image / fused embeddings on the JM benchmark:

    intra        mean cosine distance between embeddings of the same class
    inter        mean cosine distance between embeddings of different classes
    ratio        intra / inter            (lower = tighter classes)
    silhouette   sklearn silhouette score (cosine)

Usage:
    python scripts/exp_compactness.py --configs 16:0.05 16:0.07 32:0.07 32:0.10 64:0.07
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from graphformer.data import LABEL_DICT, NeuronTreeDataset, get_collate_fn
from graphformer.models import CLIPModel, CLIPLoss
from eval_knn import build_args, build_model, CKPT_CONFIG

CKPT = os.path.join(os.path.dirname(os.path.dirname(_ROOT)), "bil_act_jml_best.pth")


def compactness(z, y):
    """intra/inter cosine distance ratio + silhouette for one embedding space."""
    zn = F.normalize(z, dim=-1)
    sim = zn @ zn.t()
    dist = (1.0 - sim).clamp_min(0.0)
    y = np.asarray(y)
    n = len(y)
    same = y[:, None] == y[None, :]
    off = ~np.eye(n, dtype=bool)
    intra = dist[same & off].mean().item() if (same & off).any() else 0.0
    inter = dist[~same].mean().item() if (~same).any() else 0.0
    sil = silhouette_score(zn.numpy(), y, metric="cosine") if len(set(y)) > 1 else float("nan")
    return dict(intra=intra, inter=inter, ratio=intra / max(inter, 1e-12), silhouette=float(sil))


@torch.no_grad()
def embeddings(model, loader, device):
    model.eval()
    zt, zi, yy = [], [], []
    for batch in loader:
        batch = batch.to(device)
        zt.append(F.normalize(model.encode_tree(batch), dim=-1).float().cpu())
        zi.append(F.normalize(model.encode_image(batch.images), dim=-1).float().cpu())
        yy.append(batch.label.cpu())
    zt, zi = torch.cat(zt), torch.cat(zi)
    return zt, zi, torch.cat([zt, zi], dim=1), torch.cat(yy).numpy()


def train_one(ckpt, batch_size, temperature, epochs, lr, device, seed=42, max_steps=0,
              freeze_image_backbone=True):
    """Train the CLIP objective at one (batch size, temperature) setting.

    ``freeze_image_backbone`` is held constant across the sweep: back-propagating
    into DINOv2 needs ~9 GB at bs=8, which would make bs>8 infeasible on the
    8.6 GB RTX 4060 Laptop (see the report).  Freezing it keeps the batch-size
    effect measurable while the confound stays constant.
    """
    torch.manual_seed(seed)
    cfg = dict(CKPT_CONFIG.get(os.path.basename(ckpt), {}))
    args = build_args(temperature=temperature, **cfg)
    model, _, missing, unexpected = build_model(ckpt, args, device)
    for p in model.parameters():
        p.requires_grad_(True)
    if freeze_image_backbone:
        for p in model.image_encoder.encoder.backbone.parameters():
            p.requires_grad_(False)

    ds = NeuronTreeDataset(phase="full", dataset="JM", label_dict=LABEL_DICT["JM"],
                           image_size=518, sigma_px=16.0)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=get_collate_fn(device), num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = CLIPLoss(temperature=temperature)

    losses = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for i, batch in enumerate(loader):
            zt = model.encode_tree(batch)
            zi = model.encode_image(batch.images)
            loss = criterion(zt, zi)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
            if max_steps and i + 1 >= max_steps:
                break
        losses.append(ep_loss / max(nb, 1))
        print(f"    epoch {ep}/{epochs} loss {losses[-1]:.4f}", flush=True)

    metrics = {}
    loader_eval = DataLoader(ds, batch_size=batch_size, shuffle=False,
                             collate_fn=get_collate_fn(device), num_workers=0)
    zt, zi, zc, y = embeddings(model, loader_eval, device)
    metrics["tree"] = compactness(zt, y)
    metrics["image"] = compactness(zi, y)
    metrics["concat"] = compactness(zc, y)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
    return dict(batch_size=batch_size, temperature=temperature, epochs=epochs,
                losses=losses, metrics=metrics, seconds=time.time() - t0,
                gpu_peak_mb=peak, freeze_image_backbone=freeze_image_backbone,
                missing_keys=len(missing), unexpected_keys=len(unexpected))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, help="batch:temperature pairs, e.g. 16:0.07")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train_image_backbone", action="store_true", default=False,
                    help="back-propagate into DINOv2 (needs ~9GB at bs=8; off by default)")
    ap.add_argument("--out", default="work_dir/exp_q7_compactness.json")
    a = ap.parse_args()

    device = torch.device("cuda")
    results = []
    for spec in a.configs:
        bs, temp = spec.split(":")
        bs, temp = int(bs), float(temp)
        print(f"\n=== bs={bs} temperature={temp} epochs={a.epochs} ===", flush=True)
        try:
            r = train_one(a.ckpt, bs, temp, a.epochs, a.lr, device,
                          freeze_image_backbone=not a.train_image_backbone)
        except RuntimeError as e:
            print(f"    SKIPPED (runtime error): {e}", flush=True)
            torch.cuda.empty_cache()
            r = dict(batch_size=bs, temperature=temp, epochs=a.epochs, error=str(e)[:200])
        results.append(r)
        print(f"    ratio(concat)={r.get('metrics', {}).get('concat', {}).get('ratio')} "
              f"silhouette={r.get('metrics', {}).get('concat', {}).get('silhouette')} "
              f"peak={r.get('gpu_peak_mb')}MB", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
