"""
Persistence-image experiments (figure 4 / Q5 and figure 6 / Q8).

Q5 - the three filter functions
    f1 : radial Euclidean distance from the soma
    f2 : cumulative maximum radial distance along the soma->node path
    f3 : soma->node path length (cumulative branch length)
For each filter we report diagram statistics, the image-channel statistics, the
single-channel classification accuracy (frozen encoder) and, for the rotation
question, the stability of the resulting image and of the tree embedding under a
rotation of the neuron about the soma.

Q8 - how much morphology each channel keeps
    per-channel separability, the persistence image with 'persistence' vs
    'radius' weighting, and a direct comparison of the radius descriptors
    (mean radius used by the released pipeline vs the radius spread).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from graphformer.data import LABEL_DICT, NeuronTreeDataset, build_persistence_image, pairs_from_lines
from graphformer.data.persistence_image import filter_values
from eval_knn import build_args, build_model, CKPT_CONFIG

CKPT = os.path.join(os.path.dirname(os.path.dirname(_ROOT)), "bil_act_jml_best.pth")


def rotation_matrix_z(theta_deg):
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotate_lines(lines, R):
    out = np.array(lines, dtype=np.float64, copy=True)
    coords = out[:, 2:5]
    soma = coords[out[:, 6] == -1]
    origin = soma[0] if len(soma) else coords[0]
    out[:, 2:5] = (coords - origin) @ R.T + origin
    return out


def image_distance(a, b):
    """Relative L1 distance between two persistence-image tensors."""
    denom = a.abs().sum().item() + b.abs().sum().item()
    return float((a - b).abs().sum().item() / max(denom, 1e-12))


def channel_stats(img):
    return dict(energy=float(img.sum().item()),
                mean=float(img.mean().item()),
                max=float(img.max().item()),
                nonzero_frac=float((img > 1e-6).float().mean().item()))


def diagram_stats(pairs):
    if not pairs:
        return dict(n=0, mean_persistence=0.0, max_persistence=0.0,
                    mean_birth=0.0, mean_radius=0.0)
    pers = np.array([p["persistence"] for p in pairs])
    birth = np.array([p["birth"] for p in pairs])
    rad = np.array([p["mean_radius"] for p in pairs])
    return dict(n=len(pairs), mean_persistence=float(pers.mean()),
                max_persistence=float(pers.max()), mean_birth=float(birth.mean()),
                mean_radius=float(rad.mean()))


@torch.no_grad()
def image_features(model, images, batch=8):
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        out.append(model.image_encoder(images[i:i + batch].cuda()).float().cpu())
    return torch.cat(out)


def build_graphs(lines_list, i2ps, input_features):
    """Reuse NeuronTreeDataset tree construction on an arbitrary set of eswc arrays."""
    ds = NeuronTreeDataset.__new__(NeuronTreeDataset)      # bypass __init__
    ds.lines = [np.asarray(l)[:, input_features] for l in lines_list]
    ds.i2ps = i2ps
    return ds.process()[0]


@torch.no_grad()
def tree_features_from_graphs(model, graphs, device, batch=16):
    import dgl
    from graphformer.data import NeuronBatch

    outs = []
    for i in range(0, len(graphs), batch):
        gs = graphs[i:i + batch]
        bg = dgl.batch(gs).to(device)
        offsets, c = [0], 0
        for g in gs:
            c += g.number_of_nodes()
            offsets.append(c)
        offsets.pop(-1)
        nb = NeuronBatch(graph=bg, feats=bg.ndata["feats"].float().to(device),
                         label=torch.zeros(len(gs), dtype=torch.long).to(device),
                         offset=torch.IntTensor(offsets).to(device),
                         images=torch.zeros(len(gs), 3, 8, 8).to(device))
        outs.append(model.tree_encoder(nb).float().cpu())
    return torch.cat(outs)


def knn_acc(x, y, mask, k=20):
    """KNN accuracy with an explicit boolean train mask (dataset order is class-major)."""
    mask = np.asarray(mask)
    tr, te = np.where(mask)[0], np.where(~mask)[0]
    clf = KNeighborsClassifier(n_neighbors=min(k, len(tr)))
    clf.fit(x[tr].numpy(), y[tr].numpy())
    return clf.score(x[te].numpy(), y[te].numpy()) * 100.0


def fold_of(path):
    """Fold id parsed from ``{class}/{source}-{fold}/{file}``."""
    return int(os.path.normpath(path).replace("\\", "/").split("/")[-2].rsplit("-", 1)[-1])


def section_q5(ds, model, out):
    """Filter statistics + rotation stability."""
    lines_list = [np.asarray(l, dtype=np.float64) for l in ds.raw_lines]
    n = len(lines_list)
    train_mask = ds.train_mask

    # --- diagram / image statistics per filter -----------------------------
    per_filter = {}
    for kind in ("f1", "f2", "f3"):
        dstat, cstat = [], []
        for lines in lines_list:
            dstat.append(diagram_stats(pairs_from_lines(lines, kind)))
            img = build_persistence_image(lines, image_size=128, sigma_px=8.0, filters=(kind,))
            cstat.append(channel_stats(img))
        per_filter[kind] = dict(
            diagram={k: float(np.mean([d[k] for d in dstat])) for k in dstat[0]},
            channel={k: float(np.mean([c[k] for c in cstat])) for k in cstat[0]},
        )

    # --- single-channel classification with the frozen encoder -------------
    for kind in ("f1", "f2", "f3"):
        imgs = torch.stack([
            build_persistence_image(l, image_size=518, sigma_px=16.0, filters=(kind,)).repeat(3, 1, 1)
            for l in lines_list
        ])
        f = image_features(model, imgs)
        per_filter[kind]["knn_channel_only"] = knn_acc(f, torch.tensor(ds.targets), train_mask)

    # --- rotation stability (filters are rotation invariant by construction) --
    rot_stats = {}
    for theta in (15, 30, 45, 60, 90):
        R = rotation_matrix_z(theta)
        img_d, tree_d = [], []
        for i in range(min(n, 40)):
            lines = lines_list[i]
            rlines = rotate_lines(lines, R)
            a = build_persistence_image(lines, image_size=128, sigma_px=8.0)
            b = build_persistence_image(rlines, image_size=128, sigma_px=8.0)
            img_d.append(image_distance(a, b))
            # f-values themselves: relative change of the filter scalars
            fv = filter_values(lines[:, 2:5], lines[:, 6].astype(int), "f3")
            fv_r = filter_values(rlines[:, 2:5], rlines[:, 6].astype(int), "f3")
            tree_d.append(float(np.abs(fv - fv_r).max() / max(np.abs(fv).max(), 1e-9)))
        rot_stats[f"theta{theta}"] = dict(
            image_rel_change=float(np.mean(img_d)),
            filter_rel_change=float(np.mean(tree_d)))

    # --- rotated test-set accuracy: image branch vs tree branch ------------
    # (tree branch sees raw coordinates, so it is not rotation invariant)
    test_ds = NeuronTreeDataset(phase="test", dataset="JM", label_dict=LABEL_DICT["JM"],
                               image_size=518, sigma_px=16.0)
    train_ds = NeuronTreeDataset(phase="train", dataset="JM", label_dict=LABEL_DICT["JM"],
                                image_size=518, sigma_px=16.0)
    y = torch.tensor(train_ds.targets + test_ds.targets)
    n_train = len(train_ds.targets)
    split_mask = np.array([True] * n_train + [False] * len(test_ds.targets))

    def rotated_features(theta):
        """Image-branch and tree-branch features of a rotated neuron population."""
        imgs, rlines = [], []
        for l in list(train_ds.raw_lines) + list(test_ds.raw_lines):
            rl = rotate_lines(np.asarray(l, dtype=np.float64), rotation_matrix_z(theta))
            rlines.append(rl)
            imgs.append(build_persistence_image(rl, image_size=518, sigma_px=16.0))
        zi = image_features(model, torch.stack(imgs))
        graphs = build_graphs(rlines, train_ds.i2ps + test_ds.i2ps, train_ds.input_features)
        zt = tree_features_from_graphs(model, graphs, next(model.parameters()).device)
        return zi, zt

    rot_acc = {}
    for theta in (0, 45, 90):
        zi, zt = rotated_features(theta)
        rot_acc[f"image_theta{theta}"] = knn_acc(zi, y, split_mask)
        rot_acc[f"tree_theta{theta}"] = knn_acc(zt, y, split_mask)

    out["q5"] = dict(per_filter=per_filter, rotation=rot_stats, rotated_image_knn=rot_acc)


def section_q8(ds, model, out):
    """Channel retention + radius descriptors."""
    lines_list = [np.asarray(l, dtype=np.float64) for l in ds.raw_lines]
    y = torch.tensor(ds.targets)
    train_mask = ds.train_mask

    # --- three-channel joint vs per-channel --------------------------------
    triple = torch.stack([build_persistence_image(l, image_size=518, sigma_px=16.0) for l in lines_list])
    joint = knn_acc(image_features(model, triple), y, train_mask)
    single = {}
    for ci, kind in enumerate(("f1", "f2", "f3")):
        imgs = torch.stack([
            build_persistence_image(l, image_size=518, sigma_px=16.0, filters=(kind,)).repeat(3, 1, 1)
            for l in lines_list])
        single[kind] = knn_acc(image_features(model, imgs), y, train_mask)

    # --- persistence vs radius weighting -----------------------------------
    # 论文默认三通道加权: R(f1)=ones, G(f2)=persistence, B(f3)=radius
    weight_configs = {
        "paper_default": ["ones", "persistence", "radius"],
        "all_persistence": "persistence",
        "all_radius": "radius",
        "all_ones": "ones",
    }
    weighted = {}
    for wname, wcfg in weight_configs.items():
        imgs = torch.stack([build_persistence_image(l, image_size=518, sigma_px=16.0,
                                                     weight=wcfg)
                            for l in lines_list])
        weighted[wname] = knn_acc(image_features(model, imgs), y, train_mask)
        print(f"  q8 weight {wname}: KNN={weighted[wname]:.2f}", flush=True)

    # --- radius descriptors on their own -----------------------------------
    desc = []
    for l in lines_list:
        rad = l[:, 5]
        pers = np.array([p["persistence"] for p in pairs_from_lines(l, "f3")])
        desc.append([rad.mean(), rad.std(), rad.max(), np.median(rad),
                     pers.sum(), pers.max() if len(pers) else 0.0, len(pers)])
    desc = np.array(desc)
    scaler = StandardScaler().fit(desc[train_mask])
    X = scaler.transform(desc)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X[train_mask], y[train_mask].numpy())
    radius_knn = knn_acc(torch.tensor(X, dtype=torch.float32), y, train_mask)
    te_idx = np.where(~train_mask)[0]
    radius_logreg = clf.score(X[te_idx], y[te_idx].numpy()) * 100.0

    # per-class radius statistics (mean radius vs radius spread)
    per_class = {}
    for idx, name in ds.label_idx_to_class_names.items():
        m = y.numpy() == idx
        per_class[name] = dict(
            n=int(m.sum()),
            mean_radius=float(desc[m, 0].mean()),
            std_radius=float(desc[m, 1].mean()),
            cv_radius=float((desc[m, 1] / np.maximum(desc[m, 0], 1e-9)).mean()),
            max_radius=float(desc[m, 2].mean()),
            total_length=float(desc[m, 4].mean()),
        )

    out["q8"] = dict(joint_knn=joint, single_channel_knn=single, weighting=weighted,
                     radius_descriptors=dict(knn=radius_knn, logreg=radius_logreg,
                                             features=["mean_radius", "std_radius", "max_radius",
                                                       "median_radius", "total_persistence",
                                                       "max_persistence", "n_pairs"]),
                     per_class_radius=per_class)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default="work_dir/persistence_experiments.json")
    ap.add_argument("--sections", nargs="+", default=["q5", "q8"])
    a = ap.parse_args()

    device = torch.device("cuda")
    cfg = dict(CKPT_CONFIG.get(os.path.basename(a.ckpt), {}))
    model, meta, missing, unexpected = build_model(a.ckpt, build_args(**cfg), device)
    print(f"checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")

    ds = NeuronTreeDataset(phase="full", dataset="JM", label_dict=LABEL_DICT["JM"],
                           image_size=518, sigma_px=16.0)
    ds.train_mask = np.array([fold_of(f) < 8 for f in ds.file_list])
    print(f"neurons={len(ds)} n_train={int(ds.train_mask.sum())}")

    out = {}
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            out = json.load(f)
    if "q5" in a.sections:
        section_q5(ds, model, out)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=str)
        print("saved partial (q5) ->", a.out, flush=True)
    if "q8" in a.sections:
        section_q8(ds, model, out)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str)[:4000])
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
