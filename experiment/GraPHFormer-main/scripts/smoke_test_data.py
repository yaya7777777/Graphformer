"""Smoke test for the rebuilt graphformer.data pipeline (JML dataset)."""

import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphformer.data import (
    LABEL_DICT,
    NeuronTreeDataset,
    filter_values,
    get_collate_fn,
    pairs_from_lines,
    build_persistence_image,
)


def main():
    train = NeuronTreeDataset(phase="train", dataset="JM", label_dict=LABEL_DICT["JM"], reload=False)
    test = NeuronTreeDataset(phase="test", dataset="JM", label_dict=LABEL_DICT["JM"], reload=False)

    print("train/test:", len(train), len(test))
    print("roots:", [f.split("/")[-1] for f in train.data_dir])

    # --- single item ---
    rec = train[0]
    print("graph:", rec.graph.number_of_nodes(), "nodes,", rec.graph.number_of_edges(), "edges")
    print("images:", tuple(rec.images.shape), rec.images.dtype,
          "min/max:", float(rec.images.min()), float(rec.images.max()))
    print("label:", rec.label)

    # tree sanity: single root, every non-root node has a parent -> n-1 edges
    assert rec.graph.number_of_edges() == rec.graph.number_of_nodes() - 1

    # --- timing of image generation ---
    t = time.time()
    for i in range(min(20, len(train))):
        train[i]
    print("avg per item (graph cached + image): %.1f ms" % ((time.time() - t) / 20 * 1000))

    # --- batch ---
    loader = DataLoader(train, batch_size=4, shuffle=False, collate_fn=get_collate_fn("cpu"))
    batch = next(iter(loader))
    print("batch.graph:", batch.graph.number_of_nodes(), "nodes")
    print("batch.feats:", tuple(batch.feats.shape))
    print("batch.images:", tuple(batch.images.shape))
    print("batch.label:", batch.label.tolist(), "offset:", batch.offset.tolist())
    assert batch.feats.shape[1] == 5
    assert batch.offset[0] == 0

    # --- persistence pair stats (for the report / filter experiments) ---
    for kind in ("f1", "f2", "f3"):
        img = build_persistence_image(np.asarray(train.raw_lines[0], dtype=np.float64),
                                      image_size=128, sigma_px=16.0, filters=(kind,))
        print(f"  {kind}: image sum={float(img.sum()):.2f}")

    check_merge_tree_identity()
    check_f3_total_length_invariant(train.raw_lines)


def check_merge_tree_identity():
    """Y-shaped tree: the merge tree must yield exactly one pair, whose
    persistence equals the branch length of the shallower tip."""
    #  0 root --1--> 1 branchpoint --2--> 2 tipA, --1--> 3 tipB
    lines = np.array([
        [1, 0, 0, 0, 0, 1.0, -1],
        [2, 3, 1, 0, 0, 1.0, 1],
        [3, 3, 1, 2, 0, 1.0, 2],
        [4, 3, 1, -1, 0, 1.0, 2],
    ], dtype=np.float64)
    pairs = pairs_from_lines(lines, "f3")
    print("  Y-tree f3 pairs:", pairs)
    assert len(pairs) == 1, pairs
    assert abs(pairs[0]["persistence"] - 1.0) < 1e-9, "expected branch length 1"
    # f2 on the same tree: branchpoint radial dist 1, tipA sqrt(5), tipB sqrt(2)
    pairs_f2 = pairs_from_lines(lines, "f2")
    assert len(pairs_f2) == 1, pairs_f2
    print("  Y-tree f2 pairs:", pairs_f2)


def check_f3_total_length_invariant(raw_lines):
    """For f3 the merge tree partitions the tree: the sum of all persistences
    equals (total tree length) - (longest soma->tip path)."""
    for lines in list(raw_lines)[:10]:
        lines = np.asarray(lines, dtype=np.float64)
        parents = lines[:, 6].astype(int)
        f3 = filter_values(lines[:, 2:5], parents, "f3")
        edge = np.zeros(len(lines))
        for i, p in enumerate(parents):
            if p != -1:
                edge[i] = np.linalg.norm(lines[i, 2:5] - lines[p - 1, 2:5])
        total = edge.sum()
        tips = [i for i in range(len(lines)) if i + 1 not in set(parents[parents != -1])]
        longest = max(f3[i] for i in tips) if tips else 0.0
        s = sum(p["persistence"] for p in pairs_from_lines(lines, "f3"))
        assert abs(s - (total - longest)) < 1e-6 * max(total, 1.0), (s, total - longest)
    print(f"  f3 invariant holds on {min(10, len(raw_lines))} neurons: "
          f"sum(persistence) == total_length - longest_path")


if __name__ == "__main__":
    main()
