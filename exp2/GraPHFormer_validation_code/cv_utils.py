"""Utilities for the repository's source-fold directory layout."""
from __future__ import annotations

import re
import importlib


def make_cv_data_dir(dataset: str, test_fold: int, val_fold: int, data_root: str | None = None) -> dict[str, list[str]]:
    """Return an 8/1/1 train/val/test split from the existing 10 fold dirs."""
    if test_fold not in range(10) or val_fold not in range(10):
        raise ValueError("test_fold and val_fold must be integers in [0, 9]")
    if test_fold == val_fold:
        raise ValueError("test_fold and val_fold must be different")

    dataset_module = importlib.import_module("graphformer.data.dataset")
    old_root = dataset_module.SSL_ROOT
    if data_root:
        dataset_module.SSL_ROOT = data_root
    try:
        dirs = dataset_module.fold_dirs(dataset, range(10))
    finally:
        dataset_module.SSL_ROOT = old_root
    if not dirs:
        raise FileNotFoundError(
            f"No source-fold directories found for {dataset}. Expected "
            "data/dendrite/all_eswc_soma0_ssl/{class}/{source}-{fold}."
        )

    result = {"train": [], "val": [], "test": []}
    for path in dirs:
        match = re.search(r"-(\d+)$", path.replace("\\", "/").rsplit("/", 1)[-1])
        if not match:
            raise ValueError(f"Cannot read fold number from directory: {path}")
        fold = int(match.group(1))
        split = "test" if fold == test_fold else "val" if fold == val_fold else "train"
        result[split].append(path)

    if not result["train"] or not result["val"] or not result["test"]:
        raise ValueError(f"Incomplete fold split for {dataset}: { {k: len(v) for k, v in result.items()} }")
    return result
