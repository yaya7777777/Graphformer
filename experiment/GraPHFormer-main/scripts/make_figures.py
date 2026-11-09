"""
Plot all experiment figures from the JSON artefacts produced by
eval_knn.py / run_finetune_suite.py / exp_persistence.py / exp_compactness.py.

Usage:
    python scripts/make_figures.py --work_dir work_dir --fig_dir <deliverable>/figures
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
})

CKPTS = ["bil_act_jml_best", "glllia_weights", "N7_best_weights"]
MODES = ["tree_norm", "image_norm", "concat", "add"]
MODE_LABEL = {"tree_norm": "仅树(tree)", "image_norm": "仅图(image)",
              "concat": "拼接(concat)", "add": "相加(add)"}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fig_step1(work_dir, fig_dir):
    """Frozen-feature KNN accuracy per fusion mode and checkpoint."""
    data = {}
    for c in CKPTS:
        cv = load(os.path.join(work_dir, "eval", f"cv_{c}.json"))
        if cv:
            data[c] = cv
    if not data:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.2
    x = np.arange(len(MODES))
    for i, c in enumerate(data):
        means = [data[c]["summary"][m]["mean"] for m in MODES]
        stds = [data[c]["summary"][m]["std"] for m in MODES]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3, label=c)
    for i, m in enumerate(MODES):
        pass
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABEL[m] for m in MODES])
    ax.set_ylabel("KNN-20 准确率 (%)")
    ax.set_title("图1 步骤一：冻结特征 KNN 评估（JML 10 折交叉验证，均值±标准差）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    out = os.path.join(fig_dir, "fig1_step1_knn.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_step2(work_dir, fig_dir, exp="step2_multimodal_concat"):
    """Loss / accuracy / memory curves of the full fine-tuning run."""
    path = os.path.join(work_dir, exp, "summary.json")
    s = load(path)
    if not s or not s.get("history"):
        return None
    h = s["history"]
    ep = [r["epoch"] for r in h]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    axes[0].plot(ep, [r["loss"] for r in h], marker="o", ms=3, color="#c0392b")
    axes[0].set_xlabel("epoch"), axes[0].set_ylabel("训练损失")
    axes[0].set_title("损失曲线"), axes[0].grid(alpha=0.3)

    if any("test_acc" in r for r in h):
        acc = [(r["epoch"], r["test_acc"]) for r in h if "test_acc" in r]
        axes[1].plot([a[0] for a in acc], [a[1] for a in acc], marker="s", ms=3, color="#2471a3")
        axes[1].plot(ep, [r["train_acc"] for r in h], marker="o", ms=3, ls="--", color="#7dcea0")
        axes[1].legend(["测试准确率", "训练准确率"], fontsize=8)
    axes[1].set_xlabel("epoch"), axes[1].set_ylabel("准确率 (%)")
    axes[1].set_title("准确率"), axes[1].grid(alpha=0.3)

    axes[2].plot(ep, [r["gpu_peak_mb"] for r in h], marker="o", ms=3, color="#8e44ad")
    axes[2].set_xlabel("epoch"), axes[2].set_ylabel("显存峰值 (MB)")
    axes[2].set_title("显存占用"), axes[2].grid(alpha=0.3)

    fig.suptitle(f"图2 步骤二：完整微调（{s['mode']}/{s.get('fusion_mode')}，"
                 f"bs={s['batch_size']}，lr={s['lr']}）", y=1.04)
    out = os.path.join(fig_dir, "fig2_step2_finetune.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_ablation(work_dir, fig_dir):
    """Bar chart of the A/B/C modality ablation (plus the concat-fusion variant)."""
    suite = load(os.path.join(work_dir, "suite_main.json"))
    if not suite:
        return None
    rows = [r for r in suite["results"] if r.get("best_metric") is not None]
    if not rows:
        return None
    names = [r["name"].replace("ablation_", "").replace("step2_", "") for r in rows]
    vals = [r["best_metric"] for r in rows]
    peaks = [r.get("gpu_peak_mb") or 0 for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, vals, color=["#2471a3", "#c0392b", "#27ae60", "#8e44ad"][:len(rows)])
    ax.bar_label(bars, fmt="%.1f", fontsize=9)
    ax.set_ylabel("测试准确率 (%)")
    ax.set_ylim(0, max(vals) * 1.25 if vals else 1)
    ax.set_title("图3 步骤三／图二：模态消融（同一预训练权重，JML 4 类）")
    ax.grid(axis="y", alpha=0.3)
    for i, p in enumerate(peaks):
        ax.text(i, vals[i] * 0.45, f"{p:.0f}MB", ha="center", fontsize=8,
                color="white", fontweight="bold")
    out = os.path.join(fig_dir, "fig3_ablation_abc.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_q5(work_dir, fig_dir):
    """Filter statistics + rotation stability (image branch vs raw-coordinate branch)."""
    d = load(os.path.join(work_dir, "persistence_experiments.json"))
    if not d or "q5" not in d:
        return None
    q5 = d["q5"]
    filt = ["f1", "f2", "f3"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # diagram statistics
    x = np.arange(len(filt))
    axes[0].bar(x - 0.2, [q5["per_filter"][f]["diagram"]["n"] for f in filt], 0.4, label="配对数 n")
    axes[0].bar(x + 0.2, [q5["per_filter"][f]["diagram"]["mean_persistence"] for f in filt],
                0.4, label="平均持久性")
    axes[0].set_xticks(x), axes[0].set_xticklabels(filt)
    axes[0].legend(fontsize=8), axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_title("持久性图统计")

    axes[1].bar(x - 0.2, [q5["per_filter"][f]["knn_channel_only"] for f in filt], 0.4,
                color="#2471a3", label="单通道 KNN")
    axes[1].bar(x + 0.2, [q5["per_filter"][f]["channel"]["nonzero_frac"] * 100 for f in filt],
                0.4, color="#e67e22", label="非零像素占比(%)")
    axes[1].set_xticks(x), axes[1].set_xticklabels(filt)
    axes[1].legend(fontsize=8), axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_title("单通道可分性 / 图像占用")

    rot = q5["rotation"]
    thetas = [int(k.replace("theta", "")) for k in rot]
    axes[2].plot(thetas, [rot[k]["image_rel_change"] * 100 for k in rot], marker="o",
                 label="持久性图变化(%)")
    axes[2].plot(thetas, [rot[k]["filter_rel_change"] * 100 for k in rot], marker="s",
                 label="滤波值变化(%)")
    axes[2].set_xlabel("绕胞体旋转角 (度)"), axes[2].set_ylabel("相对变化 (%)")
    axes[2].legend(fontsize=8), axes[2].grid(alpha=0.3)
    axes[2].set_title("旋转稳定性")

    fig.suptitle("图4 图四(Q5)：三种滤波函数 f1/f2/f3", y=1.04)
    out = os.path.join(fig_dir, "fig4_q5_filters.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_q7(work_dir, fig_dir):
    d = load(os.path.join(work_dir, "exp_q7_compactness.json"))
    if not d:
        return None
    rows = [r for r in d if "metrics" in r]
    if not rows:
        return None
    labels = [f"bs={r['batch_size']}\nT={r['temperature']}" for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(rows))
    for key, color in (("tree", "#2471a3"), ("image", "#c0392b"), ("concat", "#27ae60")):
        axes[0].plot(x, [r["metrics"][key]["ratio"] for r in rows], marker="o", label=key, color=color)
        axes[1].plot(x, [r["metrics"][key]["silhouette"] for r in rows], marker="o", label=key, color=color)
    for ax, title, ylab in ((axes[0], "类内/类间距离比", "intra / inter"),
                            (axes[1], "轮廓系数", "silhouette")):
        ax.set_xticks(x), ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylab), ax.set_title(title), ax.grid(alpha=0.3), ax.legend(fontsize=8)
    fig.suptitle("图5 图五(Q7)：batch size / 温度 对类内紧度的影响", y=1.05)
    out = os.path.join(fig_dir, "fig5_q7_compactness.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_q8(work_dir, fig_dir):
    d = load(os.path.join(work_dir, "persistence_experiments.json"))
    if not d or "q8" not in d:
        return None
    q8 = d["q8"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    keys = ["joint_knn"] + [f"single:{k}" for k in q8["single_channel_knn"]]
    vals = [q8["joint_knn"]] + list(q8["single_channel_knn"].values())
    labels = ["三通道(通道保留)", "仅 f1", "仅 f2", "仅 f3"]
    bars = axes[0].bar(labels, vals, color=["#8e44ad", "#2471a3", "#27ae60", "#c0392b"])
    axes[0].bar_label(bars, fmt="%.1f", fontsize=8)
    axes[0].set_ylabel("KNN 准确率 (%)"), axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_title("三通道信息保留度")
    axes[0].tick_params(axis="x", labelsize=8)

    w = q8["weighting"]
    w_names = ["论文默认\n(per-channel)", "全 persistence", "全 radius", "全 ones"]
    w_keys = ["paper_default", "all_persistence", "all_radius", "all_ones"]
    w_vals = [w[k] for k in w_keys]
    w_colors = ["#27ae60", "#2471a3", "#e67e22", "#95a5a6"]
    bars = axes[1].bar(w_names, w_vals, color=w_colors)
    axes[1].bar_label(bars, fmt="%.1f", fontsize=8)
    axes[1].set_ylabel("KNN 准确率 (%)"), axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_title("加权策略对照")
    axes[1].tick_params(axis="x", labelsize=8)

    pc = q8["per_class_radius"]
    names = list(pc.keys())
    x = np.arange(len(names))
    axes[2].bar(x - 0.2, [pc[n]["mean_radius"] for n in names], 0.4, label="平均半径")
    axes[2].bar(x + 0.2, [pc[n]["std_radius"] for n in names], 0.4, label="半径标准差")
    axes[2].set_xticks(x), axes[2].set_xticklabels([n.replace("Isocortex_", "") for n in names],
                                                   fontsize=8, rotation=15)
    axes[2].legend(fontsize=8), axes[2].grid(axis="y", alpha=0.3)
    axes[2].set_title("各类半径描述符 (平均半径 vs 半径方差)")

    fig.suptitle("图6 图六(Q8)：三通道信息保留度 + 半径描述符对比"
                 f"（纯半径描述符 KNN {q8['radius_descriptors']['knn']:.1f}%）", y=1.04)
    out = os.path.join(fig_dir, "fig6_q8_channels.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work_dir")
    ap.add_argument("--fig_dir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)

    made = []
    for fn in (fig_step1, fig_step2, fig_ablation, fig_q5, fig_q7, fig_q8):
        try:
            p = fn(a.work_dir, a.fig_dir)
            if p:
                made.append(p)
        except Exception as e:
            print(f"  [{fn.__name__}] skipped: {type(e).__name__}: {e}")
    print("figures written:")
    for p in made:
        print("  ", p)


if __name__ == "__main__":
    main()
