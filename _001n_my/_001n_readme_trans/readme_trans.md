<div align="center">

# [CVPR 2026] 通过概念解释 CLIP 的零样本预测 (EZPC)

[![arXiv](https://img.shields.io/badge/arXiv-2603.28211-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.28211) [![Project Page](https://img.shields.io/badge/%F0%9F%8C%90%20Project-Page-blue)](https://oonat.github.io/ezpc) [![HuggingFace Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Checkpoints-FFC000)](https://huggingface.co/oonat/ezpc-checkpoints) [![HuggingFace Embeddings](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Embeddings-FFC000)](https://huggingface.co/datasets/oonat/ezpc-embeddings)

<img src="../../assets/EZPC_overview.png" width="90%" alt="Overview of EZPC"/>

**[Onat Ozdemir](https://oonat.github.io/)**<sup>*</sup> • 
**[Anders Christensen](https://scholar.google.com/citations?user=z7WhDRIAAAAJ)** • 
**[Stephan Alaniz](https://scholar.google.com/citations?user=mzZa_yQAAAAJ)** • 
**[Zeynep Akata](https://www.helmholtz-munich.de/en/eml/zeynep-akata)** • 
**[Emre Akbas](https://user.ceng.metu.edu.tr/~emre/)**

<sup>*</sup>通讯作者：`onat.ozdemir [at] ed.ac.uk`

</div>

## 新闻

- **[2026-04-09]** 🎉 我们还将在 CVPR 2026 的 **第五届可解释 AI 计算机视觉研讨会（The 5th Explainable AI for Computer Vision, XAI4CV Workshop）** 上展示 EZPC。
- **[2026-02-21]** 🎉 我们的论文被 **CVPR 2026（主会，Main）** 接收。


## 摘要

诸如 CLIP 这类大规模视觉-语言模型在零样本图像识别上取得了显著成功，然而其预测对人类理解而言在很大程度上仍然是不透明的。与之相对，概念瓶颈模型（Concept Bottleneck Models）通过基于人类定义的概念进行推理，提供了可解释的中间表示，但它们依赖于概念监督，并且缺乏向未见类别泛化的能力。我们提出 EZPC，它通过人类可理解的概念来解释 CLIP 的零样本预测，从而在这两种范式之间架起桥梁。我们的方法将 CLIP 的图文联合嵌入投影到一个从语言描述中学习得到的概念空间，从而在无需额外监督的情况下实现忠实且透明的解释。该模型通过将对齐目标（alignment objective）与重构目标（reconstruction objective）相结合来学习这一投影，确保概念激活在保持可解释性的同时保留 CLIP 的语义结构。在五个基准数据集 CIFAR-100、CUB-200-2011、Places365、ImageNet-100 和 ImageNet-1k 上的大量实验表明，我们的方法在提供有意义的概念级解释的同时，保持了 CLIP 强大的零样本分类精度。通过将开放词表（open-vocabulary）预测落地到显式的语义概念之上，我们的方法朝着可解释、可信赖的视觉-语言模型迈出了原则性的一步。

## 安装

### 环境要求

- Python 3.10
- PyTorch 2.2.0
- 支持 CUDA 的 GPU（精确复现论文报告的数值需要一块 **H100**）
- （可选）论文结果是使用 **"pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"** Docker 镜像产生的。

我们提供了一个 conda 环境文件，其中固定（pin）了产生论文结果时所用的精确软件栈：

```bash
git clone https://github.com/oonat/ezpc.git
cd ezpc
conda env create -f environment.yml
conda activate ezpc
pip install -e .
```

## 数据集准备

EZPC 在预先计算好的 CLIP/SigLIP 图像嵌入上运行。你既可以下载我们预计算好的嵌入，也可以从原始图像自行生成。

### 方案 A：下载预计算嵌入

我们将所有预计算嵌入托管在 [HuggingFace Hub](https://huggingface.co/datasets/oonat/ezpc-embeddings) 上（`huggingface-hub` 已包含在 `environment.yml` 中）：

```bash
hf download oonat/ezpc-embeddings --repo-type dataset --local-dir data
```

这会下载全部五个数据集、所有受支持骨干网络（backbone）的图像嵌入**以及**缓存的文本嵌入（`{backbone}_classname_embs.pt`、
`{backbone}_concept_matrix.pt`），可立即用于训练和评估。备齐这些文件后，`test.py` 即可精确复现论文报告的数值，
且与 GPU/CUDA 版本无关。

### 方案 B：从原始图像生成

**第 1 步。** 下载原始数据集：
```bash
python data/download_dataset.py --dataset CIFAR-100 --dataset_root ./data
```

**第 2 步。** 提取 CLIP/SigLIP 嵌入：
```bash
python data/extract_clip_features.py \
    --dataset CIFAR-100 \
    --backbone RN50 \
    --dataset_root ./data \
    --batch_size 2048 \
    --num_workers 4
```

**第 3 步。** 创建已见/未见（seen/unseen）类别划分：
```bash
python data/split_dataset.py \
    --dataset_dir ./data/CIFAR-100 \
    --backbone RN50 \
    --seed 42 \
    --ratio 0.8
```

**第 4 步。** 生成缓存的文本嵌入（类名嵌入 + 概念嵌入）：
```bash
python data/save_text_embs.py \
    --dataset CIFAR-100 \
    --dataset_root ./data \
    --backbone RN50
```

这会把 `{backbone}_classname_embs.pt` 和 `{backbone}_concept_matrix.pt` 写入
该数据集的 `embeddings/` 文件夹。`test.py` 会自动加载它们，以实现精确、
与硬件无关的复现（传入 `--recompute_text_embs` 则改为通过 CLIP 重新计算）。加上 `--overwrite` 可重新生成已存在的文件。

对每一个数据集与骨干网络的组合重复上述步骤。

<details>
<summary><b>受支持的数据集与骨干网络</b></summary>

**数据集：** `CIFAR-100`、`CUB-200-2011`、`Places365`、`ImageNet-100`、`ImageNet`

**骨干网络：** `RN50`、`ViT-B/32`、`ViT-L/14`、`ViT-SO400M-14-SigLIP-384`（以及 OpenCLIP 中其他的 CLIP/SigLIP 变体）
</details>

### 期望的数据文件夹结构

```
data/
├── CIFAR-100/
│   ├── config/
│   │   ├── cifar100_classes.txt
│   │   └── cifar100_filtered.txt
│   └── embeddings/
│       ├── {backbone}_train_embeddings.pt
│       ├── {backbone}_test_embeddings.pt
│       ├── {backbone}_classname_embs.pt
│       ├── {backbone}_concept_matrix.pt
│       ├── train_ids.pt
│       ├── test_ids.pt
│       └── splits/
│           ├── class_split.pt
│           ├── {backbone}_seen_train_embs.pt
│           ├── {backbone}_unseen_train_embs.pt
│           ├── {backbone}_seen_test_embs.pt
│           ├── {backbone}_unseen_test_embs.pt
│           ├── seen_train_ids.pt
│           ├── unseen_train_ids.pt
│           ├── seen_test_ids.pt
│           └── unseen_test_ids.pt
├── CUB-200-2011/
│   ├── config/ ...
│   └── embeddings/ ...
├── ImageNet/
│   ├── config/ ...
│   └── embeddings/ ...
├── ImageNet-100/
│   ├── config/ ...
│   └── embeddings/ ...
└── Places365/
    ├── config/ ...
    └── embeddings/ ...
```

## 用法

### 训练

学习概念投影矩阵 $A$，它把 CLIP 嵌入映射到一个可解释的概念空间：

```bash
python train.py \
    --dataset CIFAR-100 \
    --dataset_root ./data \
    --backbone RN50 \
    --lambda_weight 1.0 \
    --lr 0.01 \
    --num_epochs 10000 \
    --batch_size 1000000 \
    --device cuda
```

检查点（checkpoint）和损失曲线图默认保存到 `./checkpoints/`（可用 `--output_path` 覆盖）。

### 评估

在广义零样本分类（generalized zero-shot classification）上进行评估，并给出保真度（fidelity）指标：

```bash
python test.py \
    --dataset CIFAR-100 \
    --dataset_root ./data \
    --checkpoint_path ./checkpoints/CIFAR-100_backbone_RN50_weight_1.0_epoch_10000_lr_0.01_bs_1000000/best_A.pth \
    --backbone RN50 \
    --device cuda
```

结果（ZSL 精度、GZSL 调和平均、Top-1 一致性、Spearman 相关系数、Kendall tau、KL 散度）会保存到 `./results/`。

### 仅概念基线（A=Φ）

若要使用原始概念矩阵、不经过训练直接评估：

```bash
python test.py \
    --dataset CIFAR-100 \
    --dataset_root ./data \
    --backbone RN50 \
    --use_concept_matrix \
    --device cuda
```

## 预训练检查点

所有预训练检查点都托管在 HuggingFace 仓库 [oonat/ezpc-checkpoints](https://huggingface.co/oonat/ezpc-checkpoints) 上。下方表格中的每一行都链接到一个具体的检查点文件夹。

**要使用某个检查点，请把它的文件夹下载到仓库根目录下的 `checkpoints/` 目录中**，并保持文件夹名称不变。例如，CIFAR-100 / RN50 的检查点最终应当位于：

```
ezpc/
└── checkpoints/
    └── CIFAR-100_backbone_RN50_weight_1.0_epoch_10000_lr_0.01_bs_1000000/
        └── best_A.pth
```

你既可以点击表格中的徽章手动下载文件夹，也可以用 HuggingFace CLI 一次性拉取全部：

```bash
# 将所有检查点下载到 ./checkpoints
hf download oonat/ezpc-checkpoints \
    --local-dir . \
    --include "checkpoints/*"

# 或仅下载单个检查点文件夹
hf download oonat/ezpc-checkpoints \
    --local-dir . \
    --include "checkpoints/CIFAR-100_backbone_RN50_weight_1.0_epoch_10000_lr_0.01_bs_1000000/*"
```

下载完成后，按 [用法](#用法) 和 [实验](#实验) 章节所示，把 `--checkpoint_path` 指向相应的 `best_A.pth` 文件。

### 主要结果

| **骨干网络** | **数据集** | **检查点** | **GZSL 已见** | **GZSL 未见** | **GZSL H-Mean** |
|:--|:--|:--|:--:|:--:|:--:|
| CLIP RN50 | CIFAR-100 | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Checkpoint-FFC000.svg)](https://huggingface.co/oonat/ezpc-checkpoints/tree/main/checkpoints/CIFAR-100_backbone_RN50_weight_1.0_epoch_10000_lr_0.01_bs_1000000) | 0.365 | 0.449 | 0.403 |
| CLIP RN50 | ImageNet-100 | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Checkpoint-FFC000.svg)](https://huggingface.co/oonat/ezpc-checkpoints/tree/main/checkpoints/ImageNet-100_backbone_RN50_weight_1.0_epoch_10000_lr_0.01_bs_1000000) | 0.675 | 0.690 | 0.682 |
| CLIP RN50 | CUB-200-2011 | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Checkpoint-FFC000.svg)](https://huggingface.co/oonat/ezpc-checkpoints/tree/main/checkpoints/CUB-200-2011_backbone_RN50_weight_5.0_epoch_10000_lr_0.01_bs_1000000) | 0.457 | 0.473 | 0.465 |
| CLIP RN50 | ImageNet-1k | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Checkpoint-FFC000.svg)](https://huggingface.co/oonat/ezpc-checkpoints/tree/main/checkpoints/ImageNet_backbone_RN50_weight_1.0_epoch_10000_lr_0.01_bs_1000000) | 0.468 | 0.494 | 0.481 |
| CLIP RN50 | Places365 | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Checkpoint-FFC000.svg)](https://huggingface.co/oonat/ezpc-checkpoints/tree/main/checkpoints/Places365_backbone_RN50_weight_5.0_epoch_10000_lr_0.01_bs_1000000) | 0.339 | 0.366 | 0.352 |

### 骨干网络消融实验（ImageNet-100）

| **骨干网络** | **数据集** | **检查点** | **GZSL 已见** | **GZSL 未见** | **GZSL H-Mean** |
|:--|:--|:--|:--:|:--:|:--:|
| CLIP ViT-B/32 | ImageNet-100 | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Checkpoint-FFC000.svg)](https://huggingface.co/oonat/ezpc-checkpoints/tree/main/checkpoints/ImageNet-100_backbone_ViT-B-32_weight_1.0_epoch_10000_lr_0.01_bs_1000000) | 0.694 | 0.716 | 0.705 |
| CLIP ViT-L/14 | ImageNet-100 | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Checkpoint-FFC000.svg)](https://huggingface.co/oonat/ezpc-checkpoints/tree/main/checkpoints/ImageNet-100_backbone_ViT-L-14_weight_1.0_epoch_10000_lr_0.01_bs_1000000) | 0.812 | 0.831 | 0.822 |
| SigLIP ViT-SO400M/14 | ImageNet-100 | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Checkpoint-FFC000.svg)](https://huggingface.co/oonat/ezpc-checkpoints/tree/main/checkpoints/ImageNet-100_backbone_siglip-so400m-patch14-384_weight_1.0_epoch_10000_lr_0.01_bs_1000000) | 0.870 | 0.886 | 0.878 |

## 实验

我们提供了脚本来复现论文中的各项分析与消融实验。

### 保真度与概念干预（Faithfulness & Concept Interventions）

通过概念消融（concept ablation）评估保真度：

```bash
python experiments/faithfulness_analysis.py \
    --dataset CIFAR-100 \
    --dataset_root ./data \
    --checkpoint_path ./checkpoints/.../best_A.pth \
    --backbone RN50
```

结果会保存到 "./faithfulness_outputs" 文件夹下。

### 概念空间结构分析

生成 PCA 可视化、相似度热力图，以及激活稀疏度直方图：

```bash
python experiments/concept_space_analysis.py \
    --dataset ImageNet-100 \
    --dataset_root ./data \
    --checkpoint_path ./checkpoints/.../best_A.pth \
    --backbone RN50
```

结果会保存到 "./structure_analysis_output" 文件夹下。

### 跨数据集迁移

在源数据集上训练，并评估向目标数据集的零样本迁移：

```bash
# 训练
python experiments/cross_dataset_transfer/cross_train.py \
    --source_dataset ImageNet-100 \
    --target_dataset CUB-200-2011 \
    --dataset_root ./data \
    --backbone RN50

# 评估
python experiments/cross_dataset_transfer/cross_test.py \
    --source_dataset ImageNet-100 \
    --target_dataset CUB-200-2011 \
    --dataset_root ./data \
    --backbone RN50
```

### 定性分析

生成图像级和类别级的概念归因（concept attribution），以及基于概念的聚类可视化：

```bash
# 图像级：每张图像激活最高的概念
python experiments/qualitative_experiments/image_level_analysis.py \
    --dataset CUB-200-2011 \
    --dataset_root ./data \
    --checkpoint_path ./checkpoints/.../best_A.pth \
    --backbone RN50

# 类别级：每个类别最相关的概念
python experiments/qualitative_experiments/class_level_analysis.py \
    --dataset CUB-200-2011 \
    --dataset_root ./data \
    --checkpoint_path ./checkpoints/.../best_A.pth \
    --backbone RN50

# 基于概念的聚类
python experiments/qualitative_experiments/clustering.py \
    --dataset CUB-200-2011 \
    --dataset_root ./data \
    --checkpoint_path ./checkpoints/.../best_A.pth \
    --target_concept "has a red beak" \
    --backbone RN50
```

### 概念-区域对齐

生成 patch 级别的热力图：

```bash
python experiments/concept_region_alignment/generate_patch_heatmap.py \
    --dataset CUB-200-2011 \
    --dataset_root ./data \
    --checkpoint_path ./checkpoints/.../best_A.pth \
    --class_key "Indigo Bunting" \
    --pos_concept "a blue-gray body" \
    --neg_concept "a red face"
```

计算空间落地（spatial grounding）的 IoU 指标（仅适用于 CUB-200-2011，因为它包含分割掩码）：

```bash
python experiments/concept_region_alignment/calculate_iou_metrics.py \
    --dataset_root ./data \
    --checkpoint_path ./checkpoints/.../best_A.pth \
    --class_key "Indigo Bunting" \
    --pos_concept "a blue-gray body" \
    --neg_concept "a red face"
```

### Lambda 消融

在不同的 $\lambda$ 取值上做扫描，以研究精度-保真度（accuracy-fidelity）之间的权衡：

```bash
bash experiments/lambda_ablation/run_lambda_ablation.sh \
    --dataset ImageNet-100 \
    --dataset_root ./data \
    --lambda_values "0.01,0.1,1,10,100,1000"
```

### 词表大小消融

研究概念词表大小 $m$ 对性能的影响：

```bash
bash experiments/vocab_size_ablation/run_vocab_size_ablation.sh \
    --dataset ImageNet-100 \
    --dataset_root ./data \
    --seeds "12,123,1234" \
    --vocab_sizes "250,500,1000,2000,3000"
```

## 项目结构

```
ezpc/
├── train.py                          # 主训练脚本
├── test.py                           # 主评估脚本
├── model.py                          # EZPC 模型定义
├── utils.py                          # 工具函数、指标、数据集配置
├── dataset.py                        # 数据集类
├── environment.yml                   # Conda 环境（已固定依赖版本）
├── data/
│   ├── download_dataset.py           # 下载原始数据集
│   ├── extract_clip_features.py      # 提取 CLIP/SigLIP 图像嵌入
│   ├── split_dataset.py              # 创建已见/未见类别划分
│   ├── save_text_embs.py             # 生成缓存的类名/概念文本嵌入
│   ├── CIFAR-100/                    # 概念、类名、嵌入文件
│   ├── CUB-200-2011/
│   ├── ImageNet/
│   ├── ImageNet-100/
│   └── Places365/
└── experiments/
    ├── faithfulness_analysis.py       # 保真度与干预实验
    ├── concept_space_analysis.py      # 结构分析与 PCA
    ├── cross_dataset_transfer/        # 跨数据集实验
    ├── qualitative_experiments/       # 图像级/类别级/聚类可视化
    ├── concept_region_alignment/      # 空间落地与 IoU
    ├── lambda_ablation/               # Lambda 扫描脚本
    └── vocab_size_ablation/           # 词表大小消融
```

## 致谢

本工作中用于定义概念空间的概念词表和类别标签映射文件，最初由 [Label-free Concept Bottleneck Models](https://github.com/Trustworthy-ML-Lab/Label-free-CBM) 仓库整理并提供。我们感谢作者将这些资源开源。

## 引用

如果你觉得我们的工作有用，请引用：

```bibtex
@InProceedings{Ozdemir_2026_CVPR,
    author    = {Ozdemir, Onat and Christensen, Anders and Alaniz, Stephan and Akata, Zeynep and Akbas, Emre},
    title     = {Explaining CLIP Zero-shot Predictions Through Concepts},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {31336-31345}
}
```
