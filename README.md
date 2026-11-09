# PatchCore on MVTec AD：复现与图像级评分聚合改进

本仓库为课程期末大作业的代码与实验结果。工作内容为：

1. 完整复现 PatchCore（CVPR 2022）在 MVTec AD 15 个类别上的单类别实验；
2. 修复官方实现中分割可视化代码引用不存在属性的问题；
3. 提出一种图像级评分聚合改进：把全局最大值聚合替换为得分最高前 1% patch 的均值；
4. 给出 15 类对比实验、聚合比例敏感性分析与「缺陷稀疏度—指标失衡」的相关性分析。

---

## 1. 目录结构

```text
.
├── bin/                      # 运行入口（含 transform_std 修复）
├── src/patchcore/            # 核心实现（含 top-k 聚合补丁）
├── patches/                  # 相对上游的完整 diff
│   ├── 01-fix-transform-std.patch
│   └── 02-topk-aggregation.patch
├── scripts/
│   ├── run_baseline.sh       # 15 类基线
│   ├── run_c2.sh             # 15 类改进方法
│   ├── run_ablation.sh       # 聚合比例消融
│   └── analyze_gap.py        # 缺陷面积与指标差距的相关性分析
├── reports/                  # 各次运行的 results.csv
├── requirements.txt          # 上游依赖清单
├── requirements-lock.txt     # 本次实验的实际 pip freeze
└── README.md
```

## 2. 环境

- Python 3.8
- PyTorch 1.12.1+cu116，torchvision 0.13.1+cu116
- faiss-cpu 1.8.0
- GPU：单卡 NVIDIA RTX 3090（24 GB）即可，显存占用约 2.5 GB

### 2.1 从零创建环境

```bash
conda create -y -p ./envs/patchcore python=3.8 pip
conda activate ./envs/patchcore

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116

pip install -r requirements.txt
pip install -e .
pip install timm
```

> 上游 requirements.txt 未列出 timm，但代码实际需要。
> 若要严格对齐本次实验的包版本，可执行： pip install -r requirements-lock.txt

### 2.2 快速自检

```bash
python -c "import torch, faiss; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python bin/run_patchcore.py --help
```

第二条命令应打印出 dataset、patch_core、sampler 三个子命令。

## 3. 数据准备

从 MVTec AD 官方页面下载完整数据集（约 4.9 GB）：

https://www.mvtec.com/research-teaching/datasets/mvtec-ad

解压后的目录结构必须为：

```text
<数据根目录>/
├── bottle/
│   ├── train/good/*.png
│   ├── test/<缺陷类型>/*.png
│   └── ground_truth/<缺陷类型>/*_mask.png
├── cable/
└── ...（共 15 类）
```

即「数据根目录 / 类别 / train、test、ground_truth」三层，缺一不可。缺少 ground_truth 时无法计算像素级指标。

## 4. 复现步骤

先设置环境变量：

```bash
conda activate ./envs/patchcore
export DATA=/path/to/mvtec_ad
export OUT=/path/to/results
export CUDA_VISIBLE_DEVICES=0
```

### 4.1 基线（官方聚合方式，全局最大）

```bash
bash scripts/run_baseline.sh
```

### 4.2 本文方法（top-1% 均值聚合）

```bash
bash scripts/run_c2.sh
```

### 4.3 消融（聚合比例 ρ 的敏感性）

```bash
bash scripts/run_ablation.sh
```

### 4.4 相关性分析（缺陷面积与指标差距）

```bash
python scripts/analyze_gap.py --data "$DATA"
```

该脚本只读取 ground_truth 掩码，不修改任何文件，输出每类的平均缺陷面积占比，用于复现论文图 3。

## 5. 预期结果

单卡 3090 下每类约 3–6 分钟，15 类合计约 1 小时。

### 5.1 基线

| 指标 | 本文复现 | 原论文 |
|---|---:|---:|
| 图像级 AUROC（15 类平均） | 99.05% | 99.6% |
| 全像素级 AUROC | 98.12% | 98.1% |
| 异常像素 AUROC | 97.36% | 98.4% |

### 5.2 本文方法

| 指标 | 基线 | 本文方法 |
|---|---:|---:|
| 图像级 AUROC（15 类平均） | 99.05% | 99.12% |

逐类数值见 reports 目录下的 results.csv，每行格式为：

```text
Mean, <instance_auroc>, <full_pixel_auroc>, <anomaly_pixel_auroc>
```

> 论文中所有表格数值均直接来自 reports 中的 CSV，可逐条核对。
> 由于 GPU 型号、驱动与 FAISS 版本差异，复现结果可能与上表存在 ±0.2 个百分点的波动，属正常范围。

## 6. 方法说明

### 6.1 基线流程

1. 图像缩放到 224×224，ImageNet 归一化；
2. WideResNet50-2 提取 layer2 与 layer3 特征（512 + 1024 维）；
3. 3×3 邻域平均池化得到 patch 嵌入；
4. 贪心 coreset 采样保留 10%，构成记忆库；
5. 测试时计算每个 patch 到记忆库的最近邻距离，得到像素级异常图；
6. 图像级分数取所有 patch 分数的最大值。

### 6.2 本文改进

只改动第 6 步：取分数最高的前 K 个 patch 的均值作为图像级分数。

其中 K = max(1, ⌈ρ·P⌉)，P 为该图像的 patch 总数，ρ 取 0.01。

对应代码（src/patchcore/patchcore.py 的 score 方法）：

```python
_r = float(os.environ.get("PATCHCORE_TOPK_RATIO", "0") or 0)
if _r > 0 and x.ndim >= 1 and x.shape[-1] > 1:
    _k = max(1, int(_r * x.shape[-1]))
    x = torch.topk(x, _k, dim=-1).values.mean(dim=-1)
```

通过环境变量 PATCHCORE_TOPK_RATIO 控制：

- 不设置或设为 0：严格等价于官方实现（基线结果由此产生）
- 设为 0.01：本文方法

因此基线与改进可用同一份代码复现，二者的唯一差异就是这一个环境变量。

## 7. 相对上游的改动

完整的 diff 见 patches 目录，共两处。

### 7.1 修复可视化代码的属性引用（bin/run_patchcore.py）

官方实现的可视化代码引用了数据集中并不存在的属性 transform_std 与 transform_mean，而数据集类只定义了 IMAGENET_STD 与 IMAGENET_MEAN 常量，直接运行会抛出：

```text
AttributeError: 'MVTecDataset' object has no attribute 'transform_std'
```

本补丁将其替换为实际的 ImageNet 均值与标准差数值。

### 7.2 加入 top-k 聚合（src/patchcore/patchcore.py）

在 PatchMaker.score 方法中加入 top-k 均值聚合分支，由环境变量控制，默认行为不变。

上游基准版本：amazon-science/patchcore-inspection@fcaa92f

## 8. 权重

两个方法的 15 类记忆库（patchcore_params.pkl 与 nnscorer_search_index.faiss）已发布于本仓库的 Releases 页面：

https://github.com/xieanqi1/patchcore-mvtec-repro/releases/tag/v1.0

包含两个压缩包：

- weights_baseline.tar.gz：基线 15 类
- weights_c2.tar.gz：本文方法 15 类

说明：PatchCore 不含可训练参数，所谓「权重」实为各类别正常样本的 coreset 记忆库，因此体积较大（每套约 1 GB）。

## 9. 已知问题与说明

1. 官方参数 anomaly_scorer_num_nn 在本实现中无可观测效果：把它由 1 改为 5 后，15 个类别的像素级 AUROC 完全不变。因此本文方法只采用 top-k 聚合这一单一组件。若需复现论文结果，请保持该参数为 1。
2. 图像级分数的归一化：官方实现在计算指标前会对测试集分数做 min-max 归一化。该操作是单调变换，不改变 AUROC。
3. 随机性：seed 固定为 0；不同硬件下 FAISS 检索的浮点误差可能导致末位数字差异。

## 10. 引用与许可

```bibtex
@inproceedings{roth2022towards,
  title     = {Towards Total Recall in Industrial Anomaly Detection},
  author    = {Roth, Karsten and Pemula, Latha and Zepeda, Joaquin and Sch{\"o}lkopf, Bernhard and Brox, Thomas and Gehler, Peter},
  booktitle = {CVPR},
  year      = {2022}
}
```

本仓库基于 amazon-science/patchcore-inspection 修改，遵循上游 Apache-2.0 许可。使用 MVTec AD 数据集请遵守其官方许可（仅限非商业研究用途）。
