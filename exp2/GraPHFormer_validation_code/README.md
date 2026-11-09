# GraPHFormer 10 折验证代码

本包依据你上传的 `dataset.py`、`train.py` 和 `finetune.py` 生成。数据集代码按 `data/dendrite/all_eswc_soma0_ssl/{class}/{source}-{fold}` 读取 fold 0–9，并支持 `data_dir` 按 phase 传目录映射。验证脚本据此构造每次 8 折训练、1 折验证、1 折外层测试；每个 fold 轮流作为外层测试折，验证折取 `(test_fold + 1) % 10`。

## 放入完整工程

将以下文件放入 GraPHFormer 仓库根目录的 `cv_validation/`：

- `run_cv_validation.py`
- `train_cv.py`
- `finetune_cv.py`
- `cv_utils.py`
- `config.json`

将 `dataset_cv.py` 复制覆盖为 `graphformer/data/dataset.py`（先备份原文件）。它只增加 `val` phase，并要求验证集显式提供目录映射。

完整仓库还需包含 `graphformer/`、原始 `train.py`、`finetune.py`，以及预处理生成的 10 个 fold 数据目录。`data/info/JML_info_swc_10folds.csv` 是划分元数据；运行时 dataset 直接按生成后的 `source-0` 至 `source-9` 目录加载。`config.json` 的 `data_dir` 要指向 `all_eswc_soma0_ssl` 目录。

## 先跑一个小规模接口检查

先把 `config.json` 中 `repo_root`、`data_dir` 改成实际路径，然后从仓库根目录运行：

```powershell
python cv_validation/run_cv_validation.py --config cv_validation/config.json --stage pretrain --seeds 42 --folds 0
python cv_validation/run_cv_validation.py --config cv_validation/config.json --stage finetune --seeds 42 --folds 0 --conditions contrastive no_pretraining --modalities multimodal_concat --persistence paper_rgb
```

第一条会在 fold 0 训练、fold 1 验证、fold 0 测试。确认日志和输出正常后，再逐渐扩大运行范围。完整论文式多种子运行可使用配置中的 5 个 seeds 与 10 个 folds，但计算量很大：自监督部分是 50 次、每次 300 epochs。

## 输出

- `cv_results/checkpoints/`：每个 seed/fold 按验证 KNN 选出的自监督 checkpoint。
- `cv_results/pretrain_knn.csv`：外层测试 fold 上的冻结特征 Tree/Image/Concat/Add KNN 结果，k=5。
- `cv_results/runs.csv`：每个微调实验的验证最佳准确率、外层测试准确率、macro-F1、balanced accuracy、最佳 epoch、显存峰值和耗时。
- `cv_results/summary.csv`：按预训练方式、模态、PI 设置汇总 folds 与 seeds 的均值/标准差。
- `cv_results/logs/`：各次训练日志。

## 实验分组

默认矩阵包括：

- 预训练：对比预训练、无对比预训练（模型仍可加载 DINOv2 基础权重）。
- 模态：Tree-only、Image-only、Multimodal concat、Multimodal add。
- PI：论文三通道、f1/f2/f3 单通道、全 persistence、全 radius、全 ones。

例如，跑完整模态消融：

```powershell
python cv_validation/run_cv_validation.py --config cv_validation/config.json --stage finetune --conditions contrastive --modalities tree_only image_only multimodal_concat multimodal_add --persistence paper_rgb
```

跑持久性图通道对照：

```powershell
python cv_validation/run_cv_validation.py --config cv_validation/config.json --stage finetune --conditions contrastive --modalities multimodal_concat --persistence paper_rgb f1_only f2_only f3_only all_persistence all_radius all_ones
```

## 目前未实现的对照

错配树–图配对的 InfoNCE 需要改 `graphformer/models/clip_model.py` 的 loss 接口；半径 CV/path-slope 需要改 `graphformer/data/persistence_image.py`。你目前提供了这两个文件的目录截图，但没有源码，因此没有在运行器里假设内部函数接口或伪造结果。

## 显存

配置按论文自监督设置使用 batch 128、图像 112×112、300 epochs。该设置参考论文 RTX 4090 运行环境；8 GB 显存不能保证原样运行。下游 batch 8 是按你已有复现配置填写，因为论文附件没有完整给出监督微调参数。
