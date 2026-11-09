"""
Neuron dataset for GraPHFormer.

Rebuilds the ``graphformer.data`` package that the released repository omits
(``finetune.py`` imports ``NeuronTreeDataset``, ``get_collate_fn``, ``LABEL_DICT``
from it).  The tree part follows the TreeMoCo ``tree_dataset.py`` implementation
(which is the codebase this model's data pipeline descends from); the image part
is the persistence image generator in ``persistence_image.py``.

A dataset item is a ``NeuronRecord(graph, images, label)`` where ``graph`` is a
DGL tree whose node features are ``lines[:, input_features]``.  The collate
function returns a ``NeuronBatch`` exposing exactly the four attributes the
models expect - ``graph``, ``feats``, ``label``, ``offset`` - plus ``images``.

Folder layout expected under ``data/dendrite/all_eswc_soma0_ssl``::

    {class}/{source}-{fold}/{neuron}.swc

with ``source`` in {``bil``, ``allen_cell_type``, ``janelia_mouselight``} and
``fold`` in 0..9 (folds 0-7 train, 8-9 test, following ``split_BIL_JM_ACT``).
Which neurons exist is driven entirely by the CSVs in ``data/info``.
"""

import hashlib
import os
import pickle
import re
from collections import namedtuple

import dgl
import networkx as nx
import numpy as np
import torch
from dgl.convert import from_networkx
from nltk import tree

from .persistence_image import build_persistence_image

# ---------------------------------------------------------------------------
# class definitions (identical to the released TreeMoCo/NeuronRepresentation)
# ---------------------------------------------------------------------------

all_wo_others = {
    "VPM": 0,
    "Isocortex_layer23": 1,
    "Isocortex_layer4": 2,
    "PRE": 3,
    "SUB": 4,
    "CP": 5,
    "VPL": 6,
    "Isocortex_layer6": 7,
    "MG": 8,
    "Isocortex_layer5": 9,
}

bil_6_classes = {
    "CP": 0,
    "Isocortex_layer23": 1,
    "Isocortex_layer4": 2,
    "Isocortex_layer5": 3,
    "Isocortex_layer6": 4,
    "VPM": 5,
}

# the sparse MouseLight subset of the MERFISH cell-type space (4 classes)
JM_classes = {
    "Isocortex_layer23": 0,
    "Isocortex_layer5": 1,
    "Isocortex_layer6": 2,
    "VPM": 3,
}

# the Allen Cell Types subset (4 classes)
ACT_classes = {
    "Isocortex_layer23": 0,
    "Isocortex_layer4": 1,
    "Isocortex_layer5": 2,
    "Isocortex_layer6": 3,
}

LABEL_DICT = {
    "all_wo_others": all_wo_others,
    "bil_6_classes": bil_6_classes,
    "JM": JM_classes,
    "ACT": ACT_classes,
}

#: metric name -> source folder prefix used in ``all_eswc_soma0_ssl``
DATASET_SOURCE = {
    "bil_6_classes": "bil",
    "JM": "janelia_mouselight",
    "ACT": "allen_cell_type",
    "all_wo_others": None,          # mixes every source
}

SSL_ROOT = os.path.join("data", "dendrite", "all_eswc_soma0_ssl")
CACHE_ROOT = os.path.join("data", "dendrite", "processed_datasets")

#: minimum number of branch nodes for a neuron to be kept
MIN_NODES = 10


# ---------------------------------------------------------------------------
# batches
# ---------------------------------------------------------------------------

class NeuronRecord(namedtuple("NeuronRecord", ["graph", "images", "label"])):
    """One dataset item."""

    __slots__ = ()

    def to(self, device):
        return NeuronRecord(self.graph.to(device), self.images.to(device), self.label.to(device))


class NeuronBatch(namedtuple("NeuronBatch", ["graph", "feats", "label", "offset", "images"])):
    """A collated batch; ``graph``/``feats`` are DGL-side, ``images`` torch-side."""

    __slots__ = ()

    def to(self, device):
        return NeuronBatch(
            graph=self.graph.to(device),
            feats=self.feats.to(device),
            label=self.label.to(device),
            offset=self.offset.to(device),
            images=self.images.to(device),
        )


def get_collate_fn(device, use_images=True):
    """Build the collate function producing a :class:`NeuronBatch` on ``device``."""

    def collate_fn(batch):
        trees, offsets, labels, images = [], [0], [], []
        cnt = 0
        for graph, img, label in batch:
            trees.append(graph)
            cnt += graph.number_of_nodes()
            offsets.append(cnt)
            labels.append(label)
            images.append(img)
        offsets.pop(-1)

        batch_trees = dgl.batch(trees)
        return NeuronBatch(
            graph=batch_trees.to(device),
            feats=batch_trees.ndata["feats"].float().to(device),
            label=torch.as_tensor(labels).to(device),
            offset=torch.IntTensor(offsets).to(device),
            images=torch.stack(images).to(device),
        )

    return collate_fn


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

def fold_dirs(dataset, folds):
    """Existing ``all_eswc_soma0_ssl/{class}/{source}-{fold}`` directories."""
    source = DATASET_SOURCE.get(dataset)
    classes = list((LABEL_DICT[dataset] if dataset in LABEL_DICT else {}).keys())
    if not classes:                     # e.g. data_bil_3class
        classes = sorted(os.listdir(SSL_ROOT)) if os.path.isdir(SSL_ROOT) else []
    sources = [source] if source is not None else [
        DATASET_SOURCE[k] for k in ("bil_6_classes", "JM", "ACT")
    ]
    dirs = []
    for cls in classes:
        for src in sources:
            for fold in folds:
                candidate = os.path.join(SSL_ROOT, cls, f"{src}-{fold}")
                if os.path.isdir(candidate):
                    dirs.append(candidate)
    return dirs


class NeuronTreeDataset(torch.utils.data.Dataset):
    """Morphology dataset returning (tree graph, persistence image, label)."""

    def __init__(
        self,
        phase="train",
        dataset="JM",
        label_dict=None,
        input_features=(2, 3, 4, 12, 13),
        topology_transformations=None,
        attribute_transformations=None,
        data_dir=None,
        use_images=True,
        image_size=518,
        persistence_augmentation=None,
        sigma_px=16.0,
        persistence_filters=("f1", "f2", "f3"),
        persistence_weight="persistence",
        balance=False,
        reload=False,
        cache_images=False,
    ):
        assert phase in ["train", "val", "test", "full"], phase
        assert dataset in LABEL_DICT or dataset in ["data_bil_3class"], dataset

        self.phase = phase
        self.dataset = dataset
        self.input_features = list(input_features)
        self.topology_transformations = topology_transformations
        self.attribute_transformations = attribute_transformations
        self.use_images = use_images
        self.image_size = image_size
        self.persistence_augmentation = persistence_augmentation
        self.sigma_px = sigma_px
        self.persistence_filters = tuple(persistence_filters)
        self.persistence_weight = persistence_weight
        self.balance = balance
        self.reload = reload
        self.cache_images = cache_images
        self._image_cache = {}

        self.label_dict = label_dict if label_dict is not None else LABEL_DICT[dataset]
        self.classes = list(self.label_dict.keys())
        self.label_idx_to_class_names = {v: k for k, v in self.label_dict.items()}

        self.data_dir = self._resolve_data_dir(data_dir)
        self.load_filecache()

        print(f"[{self.dataset}/{self.phase}] n={len(self.targets)}", flush=True)
        print(
            f"[{self.dataset}/{self.phase}] label distribution:",
            dict(zip(*[list(x) for x in np.unique(self.targets, return_counts=True)])),
            flush=True,
        )

        # precompute graphs when no augmentation is active (TreeMoCo fast path)
        if topology_transformations is None and attribute_transformations is None:
            self.neuron_trees, self.tree_lens = self.process()
        else:
            self.neuron_trees, self.tree_lens = None, None

    # -- data location -----------------------------------------------------

    def _resolve_data_dir(self, data_dir):
        """Return the list of fold directories for this dataset/phase."""
        if isinstance(data_dir, dict):
            return data_dir[self.phase]
        if self.phase == "val":
            raise ValueError("phase='val' requires an explicit data_dir mapping with a 'val' entry")
        folds = {"train": range(0, 8), "test": range(8, 10), "full": range(0, 10)}[self.phase]
        dirs = fold_dirs(self.dataset, folds)
        if not dirs:
            raise FileNotFoundError(
                f"No data directories found for dataset='{self.dataset}' phase='{self.phase}'. "
                f"Expected {SSL_ROOT}/{{class}}/{{source}}-{{fold}}. "
                f"Run the preprocessing pipeline first (scripts/prepare_jml_data.py)."
            )
        return dirs

    # -- caching -----------------------------------------------------------

    def _cache_tag(self):
        """Deterministic fingerprint of the selected fold directories.

        Keeps the on-disk cache race-free when several runs (e.g. the 10 CV folds)
        share the same dataset/phase name.
        """
        key = "|".join(sorted(os.path.normpath(d).replace("\\", "/") for d in self.data_dir))
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def load_filecache(self):
        os.makedirs(CACHE_ROOT, exist_ok=True)
        cache_path = os.path.join(CACHE_ROOT, f"{self.dataset}_{self.phase}_{self._cache_tag()}.pkl")

        if os.path.exists(cache_path) and not self.reload:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            lines, i2ps, targets, file_list = (
                cached["lines"], cached["i2ps"], cached["targets"], cached["file_list"]
            )
        else:
            lines, i2ps, targets, file_list = self.load_and_cache(cache_path)

        # keep the full eswc rows (n,type,x,y,z,radius,parent,...) for image
        # generation, and the feature-sliced rows for the tree graphs
        self.raw_lines = lines
        self.lines = [x[:, self.input_features] for x in lines]
        self.i2ps, self.targets, self.file_list = i2ps, targets, file_list

    def load_and_cache(self, cache_path):
        file_list, lines, i2ps, targets = [], [], [], []
        for filename in self._list_file():
            data = self.readswc(filename)
            file_list.append(filename)
            lines.append(data)
            i2ps.append({int(p[0]): int(p[6]) for p in data})
            targets.append(self._label_from_path(filename))

        with open(cache_path, "wb") as f:
            pickle.dump(
                {"lines": lines, "i2ps": i2ps, "targets": targets, "file_list": file_list}, f
            )
        print(f"saved cache -> {cache_path}", flush=True)
        return lines, i2ps, targets, file_list

    def _list_file(self):
        """Flat listing of the fold directories, keeping neurons with >= MIN_NODES nodes."""
        files, per_class = [], {}
        for d in self.data_dir:
            for name in os.listdir(d):
                path = os.path.join(d, name)
                if not os.path.isfile(path) or not name.endswith(("swc", "eswc")):
                    continue
                with open(path) as f:
                    n = sum(1 for line in f if not line.startswith("#"))
                if n < MIN_NODES:
                    continue
                files.append(path)
                per_class.setdefault(self._label_from_path(path), []).append(path)

        if self.balance and self.phase == "train":
            m = max(len(v) for v in per_class.values())
            expanded = []
            for v in per_class.values():
                extra = np.random.choice(v, size=m - len(v), replace=True)
                expanded.extend(list(v) + list(extra))
            files = expanded

        print(f"{self.dataset}/{self.phase}: {len(files)} neurons", flush=True)
        return files

    def _label_from_path(self, filename):
        """Class folder is ``.../{class}/{source}-{fold}/{file}`` -> index -3."""
        parts = filename.replace("\\", "/").split("/")
        for depth in (3, 4):
            if len(parts) >= depth and parts[-depth] in self.label_dict:
                return self.label_dict[parts[-depth]]
        raise KeyError(f"cannot infer class from {filename}")

    def readswc(self, filename):
        with open(filename) as f:
            lines = f.read().strip().split("\n")
        i = 0
        while lines[i].startswith("#"):
            i += 1
        lines = lines[i:]

        for i in range(len(lines)):
            lines[i] = re.split(r"[\[\],\s]", lines[i])
            while "" in lines[i]:
                lines[i].remove("")
        return np.array(lines).astype(float)

    # -- tree construction -------------------------------------------------

    def process(self):
        neuron_trees, tree_lens = [], []
        for i, line in enumerate(self.lines):
            root = self.lines2trees(line, self.i2ps[i])
            _tree, tree_len = self._build_tree(root)
            neuron_trees.append(_tree)
            tree_lens.append(tree_len)
        return neuron_trees, tree_lens

    def lines2trees(self, lines, i2p):
        p2t = dict()
        root = None
        for i in i2p:
            if int(i2p[i]) == -1:
                root = tree.Tree(lines[int(i) - 1], [])
                p2t[i] = root
                break

        def add_node(i, line):
            temp = tree.Tree(line, [])
            if int(i2p[i]) not in p2t:
                add_node(int(i2p[i]), lines[int(i2p[i]) - 1])
            p2t[int(i2p[i])].append(temp)
            p2t[i] = temp

        for i, line in enumerate(lines):
            if int(i2p[i + 1]) == -1:
                continue
            if i + 1 not in p2t:
                add_node(i + 1, line)
        return root

    def _build_tree(self, root):
        g = nx.DiGraph()

        def _rec_build(nid, node):
            for child in node:
                cid = g.number_of_nodes()
                g.add_node(cid, feats=child.label())
                if len(child) != 0:
                    _rec_build(cid, child)
                g.add_edge(cid, nid)

        g.add_node(0, feats=root.label())
        _rec_build(0, root)
        ret = from_networkx(g, node_attrs=["feats"])
        return ret, ret.number_of_nodes()

    # -- items -------------------------------------------------------------

    def _image(self, idx):
        if self.cache_images and idx in self._image_cache:
            return self._image_cache[idx]
        img = build_persistence_image(
            np.asarray(self.raw_lines[idx], dtype=np.float64),
            image_size=self.image_size,
            sigma_px=self.sigma_px,
            filters=self.persistence_filters,
            weight=self.persistence_weight,
            augmentation=self.persistence_augmentation,
        )
        # DINOv2 expects 3 input channels; if the persistence image has fewer
        # channels (e.g. a single f2 filter), tile/repeat along the channel axis.
        c = img.shape[0]
        if c < 3:
            img = img.repeat(3 // c + (1 if 3 % c else 0), 1, 1)[:3]
        elif c > 3:
            img = img[:3]
        if self.cache_images:
            self._image_cache[idx] = img
        return img

    def __getitem__(self, idx):
        if self.neuron_trees is not None:
            graph, label = self.neuron_trees[idx], self.targets[idx]
        else:
            lines = self.lines[idx].copy()
            if self.attribute_transformations is not None:
                lines = self.attribute_transformations(lines)
            neuron = self.lines2trees(lines, self.i2ps[idx])
            if self.topology_transformations is not None:
                neuron = self.topology_transformations(neuron)
            graph, _ = self._build_tree(neuron)
            label = self.targets[idx]

        images = self._image(idx) if self.use_images else torch.zeros(1)
        return NeuronRecord(graph=graph, images=images, label=label)

    def __len__(self):
        return len(self.targets)
