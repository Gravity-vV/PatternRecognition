# 模式识别系统开发文档：多模块化评测框架 (PR-Eval-Framework)

## 1. 目标与范围 (Goals & Scope)

**核心目标**：构建一个高度模块化的模式识别流水线（Pipeline），支持灵活组合不同的预处理、特征提取和分类器，并通过自动化实验找到针对特定数据集的最佳组合。

**对比目标（传统 vs 前沿）**：在同一数据集上同时提供两条可对比基线，并输出统一指标与耗时：

- **传统基线**：手工特征 + 传统分类器（如 HOG/PCA + SVM/RF）。
- **现代基线**：预训练视觉模型迁移学习（fine-tune）或冻结表征（embedding）+ 线性分类器。

**约束**：面向 Apple Silicon（例如 M4 / 16GB），训练总时长目标为 **1～2 小时内完成**（或使用已训练权重直接评测）。

---

## 2. 数据集选择与算力假设 (Dataset & Compute)

### 2.1 主推荐数据集

- **Oxford-IIIT Pets**（37 类，约 7k 图片，真实照片、细粒度）：
  - 适合对比：传统方法与迁移学习差异明显，报告可解释性强。
  - 训练建议：输入统一到 224×224；优先使用 PyTorch `mps`。

### 2.2 可选数据集（更难）

- CIFAR-100（100 类，32×32）
- Tiny-ImageNet（200 类，64×64，训练更久）

---

## 3. 系统架构 (Architecture)

系统采用**插件式架构 (Plugin Architecture)**，由五个核心组件构成：

1. **Data Loader**：数据加载与划分（train/val/test），并提供统一的 `(X, y)` 输出。
2. **Preprocessor**：预处理模块（可链式/多步）。
3. **Feature Extractor**：特征提取模块（传统/现代表征）。
4. **Classifier**：分类决策模块（传统 ML 或深度模型训练器）。
5. **Evaluator**：性能分析与可视化模块。

**统一接口约定**：所有模块继承统一基类（例如 `BaseModule`），遵循：

- `fit(X, y=None, **kwargs)`
- `transform(X, **kwargs)`（用于预处理/特征提取）
- `predict(X, **kwargs)`（用于分类器）

---

## 4. 模块规格 (Modules)

> **Agent 指令**：请按照以下接口规范实现各个类。每个类需继承统一基类，保证可在实验脚本中被“组合/替换”。

### 4.1 传统流水线模块（Classic Pipeline）

#### A. 预处理 (Preprocessing)

支持链式调用。

* **Methods**:
1. `GrayScaler`：灰度化处理。
2. `Normalizer`：归一化（Min-Max 或 Z-Score）。
3. `DenoisingFilter`：降噪（中值滤波、高斯滤波）。
4. `Augmentor`：传统增强（随机旋转、平移；仅用于训练集）。

#### B. 特征提取 (Feature Extraction)

* **Methods**:
1. `PixelFlattener`：原始像素展开。
2. `HOGExtractor`：方向梯度直方图（形状/边缘特征）。
3. `LBPExtractor`：局部二值模式（纹理特征）。
4. `PCAExtractor`：主成分分析（降维特征）。

#### C. 分类决策 (Classification)

* **Methods**:
1. `SVMClassifier`：支持向量机（核函数可选）。
2. `RandomForestClassifier`：随机森林。
3. `KNNClassifier`：K-近邻。
4. `MLPClassifier`：多层感知机（轻量级神经网络）。

### 4.2 现代视觉模块（Modern Baselines）

> 目标：引入“前沿可落地”的基线，并确保在 Mac（MPS）上 1～2 小时可完成训练或使用预训练权重直接评测。

#### D. 现代预处理 (Vision Preprocessing)

* **Methods（建议补充）**:
1. `Resize`：统一尺寸（如 224×224）。
2. `ImageNetNormalizer`：按 ImageNet mean/std 归一化（与传统 `Normalizer` 区分）。
3. `TorchAugmentor`：训练集增强（RandomResizedCrop / HorizontalFlip 等）。

#### E. 现代特征/分类 (Modern Feature & Classifier)

* **Methods（至少实现 2 类，便于对比）**:
1. `TorchTransferClassifier`：迁移学习分类器（PyTorch + `torchvision` 或 `timm` 的 ImageNet 预训练权重）。
  - 推荐骨干：`convnext_tiny`（CNN，非 transformer）。
   - 训练模式：
     - baseline A：冻结 backbone，仅训练分类头（快、对比清晰）。
     - baseline B：解冻后若干层 fine-tune（更强、1～2 小时级）。
2. `FrozenEmbeddingClassifier`：冻结预训练表征 + 线性分类器。
   - 流程：`EmbeddingExtractor(frozen)` -> `LinearClassifier(LogReg/LinearSVC)`。
   - 优点：训练极快，适合把“前沿表征”接入传统评测框架做横向对比。

---

## 5. 实验设计与输出规范 (Experiments & Outputs)

### 5.1 组合遍历逻辑（ExperimentRunner）

> **Agent 指令**：实现 `ExperimentRunner`，能够遍历并评测不同组合。

**传统组合**（示例）：

- `PreprocessorChain`（可选多步） × `FeatureExtractor`（单选） × `Classifier`（单选）

**现代组合**（示例）：

- `VisionPreprocess` × `TorchTransferClassifier(mode=A/B)`
- `VisionPreprocess` × `FrozenEmbeddingClassifier(embedding=convnext_tiny, head=LogReg/LinearSVC)`

**复现要求**：固定随机种子、固定数据划分策略，统一输出字段。

### 5.2 评价指标 (Metrics)

每个组合至少输出：

- **Accuracy / Precision / Recall / F1-Score**
- **Training Time / Inference Time**
- **Confusion Matrix**（混淆矩阵）

### 5.3 结果输出（CSV Schema）

建议 CSV 字段（可扩展但需稳定）：

- `dataset`
- `split_seed`
- `preprocess`
- `feature`
- `classifier`
- `accuracy` `precision` `recall` `f1`
- `train_time_sec` `inference_time_sec`
- `n_train` `n_test`
- `model_path`（如保存权重/模型文件）

### 5.4 对比最低要求（传统 vs 现代）

- 同一数据集、同一划分（固定随机种子）下，至少输出两组对比：
  - 传统：`HOG + SVM`、`PCA + RandomForest`（或你实现的传统组合）
  - 现代：`Transfer(convnext_tiny) + LinearHead`、`FrozenEmbedding + LinearClassifier`

---

## 6. 开发路线图 (Task Prompts)

### 任务 1：搭建骨架与统一接口（Foundation）

**目标**：把“流水线可组合”这件事先打牢。

**产出**：

- 目录结构：`data/`, `modules/`, `experiments/`, `utils/`
- 抽象基类：`BaseModule`（统一 `fit/transform/predict` + 可选 `fit_transform`）
- 最小自检：能 import 各包并跑通一个 dummy 流水线

**验收**：`python -m experiments.smoke_test` 可运行且通过。

### 任务 2：实现传统模式识别模块（Classic Modules）

**目标**：先把“传统流程”完整跑通，作为对比基线。

**产出**（在 `modules/` 下）：

- 预处理：高斯滤波、数值归一化（Min-Max / Z-Score）
- 特征：HOG、PCA
- 分类器：SVM、随机森林（sklearn 封装）

**验收**：给定一个小样本数据（可用内置数据集），能够完成：预处理 → 特征 → 训练 → 预测。

### 任务 3：跑通传统流水线的 Runner（Classic Runner → CSV）

**目标**：把“实验自动化”和“结果可复现”做出来（先覆盖传统组合）。

**产出**：

- `ExperimentRunner`（或等价脚本）：遍历传统组合 `Preprocess(可选) → Feature → Classifier`
- 统一结果 CSV：严格遵循第 5.3 节 schema（至少含 `dataset/split_seed/preprocess/feature/classifier/accuracy/train_time_sec/inference_time_sec`）
- 固定随机种子 + 固定划分策略（保证可复现）

**验收**：一次运行能产出 CSV，且至少包含 2 个传统组合（例如 `HOG+SVM`、`PCA+RF`）。

### 任务 4：实现前沿流程并接入同一 Runner（Modern Baselines → 同 schema CSV）

**目标**：在不破坏传统 Runner 的前提下，引入“前沿但可落地”的现代基线，并输出与传统相同 schema 的结果，方便后续统一可视化。

**产出**：

- 数据：新增 `data/` 加载器支持 Oxford-IIIT Pets（或 CIFAR-100）
- 现代模块（在 `modules/` 下）：
  - `TorchTransferClassifier`：预训练 `convnext_tiny` 迁移学习（至少支持“冻结头训练”；可选支持“部分 fine-tune”）
  - `FrozenEmbeddingClassifier`：冻结预训练表征（embedding）+ 线性分类器（LogReg/LinearSVC）
  - 现代预处理：Resize(224) + ImageNet mean/std + 训练增强（仅训练集）
- Runner 扩展：在同一数据集/同一划分下，同时跑传统与现代组合，输出到同一 CSV（或多个 CSV 但 schema 一致）
- 复现：保存模型/权重文件，并在 CSV 中记录 `model_path`

**验收**：在 Apple Silicon（MPS）上可完成至少 2 个现代组合的训练/评测，并与传统结果同表可对比。

### 任务 5：统一对比、可视化与可选交互入口（Compare & Visualize）

**目标**：把“结果解释与展示”补齐，让传统与前沿的差异一眼可见。

**产出**：读取符合第 5.3 节 schema 的 CSV（传统/现代均可），输出：

- 柱状图：按 `feature`/`classifier`（含现代基线）对比 accuracy（或其它指标）
- 热力图：横 `feature`，纵 `classifier`，单元格为 accuracy
- 混淆矩阵：自动挑选最好与最差组合各画一张并做简要分析

**可选加分（交互入口，二选一即可）**：

- CLI：参数选择数据集/流程/组合，触发训练与评测并输出 CSV
- 轻量 Web：例如 Streamlit，支持选择组合并触发训练/评测

**验收**：给定一份包含传统与现代结果的 CSV，能一键生成上述图表与混淆矩阵。

---

## 7. 加分项 (Bonus Points)

1. **交叉验证 (Cross-Validation)**：使用 5-fold CV，而非单次 train/test。
2. **消融实验 (Ablation Study)**：例如去掉降噪或去掉增强，量化性能下降。
3. **结果解读**：给出对比解释（例如为什么 HOG+SVM 可能优于 PCA+KNN，或为什么迁移学习显著提升）。

