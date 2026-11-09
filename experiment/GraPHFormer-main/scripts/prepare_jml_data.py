"""
JML-only data preparation.

The released prepare_data.py requires allensdk and the BIL/ACT raw data.
This driver reproduces the same steps for the Janelia MouseLight (JML) dataset,
which is the only public dataset available in this environment:

    json  -> swc -> swc_soma0 (normalize) -> dendrite swc_soma0 (drop axon)
          -> eswc_soma0 (branch features) -> all_eswc_soma0_ssl folder tree

The metadata csv (data/info/JML_info_swc.csv) and the 10-fold split csv
(data/info/JML_info_swc_10folds.csv) are already provided in the repo, so the
allensdk-dependent `convert_janelia_json` metadata step and the
`split_sample_10fold_cv_and_merge` step are skipped here.
"""

import glob
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_ROOT = Path(os.path.abspath(__file__)).parent
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT / "data_io"))

from data_io.neuron import Neuron  # noqa: E402
from data_io.process_raw_script import (  # noqa: E402
    filter_axon_and_check,
    normalize_root_and_check,
    summarize_branch,
)

DATA_PATH = "./data"
SOURCE = "janelia_mouselight"
# class folders we actually need for the JM 4-class task
JM_CLASSES = ["Isocortex_layer23", "Isocortex_layer5", "Isocortex_layer6", "VPM"]
ACRONYM_ALIASES = {"Isocortex_layer2/3": "Isocortex_layer23"}


def json_to_swc(folder_in, folder_out):
    """Janelia MouseLight json -> swc (no metadata csv needed)."""
    os.makedirs(folder_out, exist_ok=True)
    filelist = sorted(glob.glob(folder_in))
    print(f"converting {len(filelist)} json files -> swc")
    for fname_json in tqdm(filelist):
        fname_swc = os.path.join(folder_out, os.path.basename(fname_json).replace(".json", ".swc"))
        neuron = Neuron()
        neuron.load_json(fname_json)
        neuron.save_swc(fname_swc)


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil

        shutil.copyfile(src, dst)


def build_class_folders(data_path, source, classes, acronym_aliases, csv_rel):
    """Reproduce the 'hack folder creation' block of prepare_data.py for one dataset."""
    csv = pd.read_csv(os.path.join(data_path, csv_rel))
    n_ok, n_missing, n_skip = 0, 0, 0
    for split in range(10):
        fold_rows = csv[csv["model__fold"] == split]
        for _, row in fold_rows.iterrows():
            fname = row["swc__fname"]
            acronym = row["structure_merge__acronym"]
            acronym = acronym_aliases.get(acronym, acronym)
            if acronym not in classes:
                n_skip += 1
                continue
            target_dir = os.path.join(
                data_path, "dendrite", "all_eswc_soma0_ssl", acronym, f"{source}-{split}"
            )
            os.makedirs(target_dir, exist_ok=True)
            src = os.path.abspath(os.path.join(data_path, "dendrite", source, "eswc_soma0", fname))
            if os.path.isfile(src):
                link_or_copy(src, os.path.join(target_dir, fname))
                n_ok += 1
            else:
                n_missing += 1
                print(f"missing: {src}")
    print(f"fold folders: linked={n_ok}, missing={n_missing}, skipped(unused class)={n_skip}")


if __name__ == "__main__":
    # 1) json -> swc
    json_to_swc(
        os.path.join(DATA_PATH, "raw", SOURCE, "json30", "*.json"),
        os.path.join(DATA_PATH, "raw", SOURCE, "swc"),
    )

    # 2) normalize (root to origin + dendrite PCA rotation + type override)
    normalize_root_and_check(
        os.path.join(DATA_PATH, "raw", SOURCE, "swc", "*.swc"),
        os.path.join(DATA_PATH, "raw", SOURCE, "swc_soma0"),
    )

    # 3) drop axon, keep dendrite
    filter_axon_and_check(
        os.path.join(DATA_PATH, "raw", SOURCE, "swc_soma0", "*.swc"),
        os.path.join(DATA_PATH, "dendrite", SOURCE, "swc_soma0"),
    )

    # 4) summarize branches -> eswc with branch features
    summarize_branch(
        os.path.join(DATA_PATH, "dendrite", SOURCE, "swc_soma0", "*.swc"),
        os.path.join(DATA_PATH, "dendrite", SOURCE, "eswc_soma0"),
    )

    # 5) 10-fold class folder structure
    build_class_folders(
        DATA_PATH, SOURCE, JM_CLASSES, ACRONYM_ALIASES, os.path.join("info", "JML_info_swc_10folds.csv")
    )
    print("JML data preparation done.")
