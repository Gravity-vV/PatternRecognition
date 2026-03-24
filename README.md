# PatternRecognition（PR-Eval-Framework）

以 `devdoc.md` 为规范实现的模块化评测框架（当前 Web/Runner 默认使用 Oxford-IIIT Pets 数据集）。

## 环境

```zsh
cd /Users/hh/MacCodes/PatternRecognition
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 自测

```zsh
source .venv/bin/activate
python -m experiments.smoke_test
python -m experiments.task2_smoke_test
python -m experiments.task4_smoke_test
python -m experiments.task5_smoke_test
```

## Task 4：Pets（传统+现代）同 schema CSV

注意：首次运行会下载 Oxford-IIIT Pets 数据集；ConvNeXt-Tiny 预训练权重会缓存复用（不会重复下载）。

缓存位置：
- 数据集：`data/oxford_pets/`
- torchvision 预训练权重：`.cache/torch/`（通过 `TORCH_HOME` 固定到项目内）

```zsh
source .venv/bin/activate
python -m experiments.task4_runner --split-seed 0 --epochs 1 --batch-size 32 \
  --out-csv experiments/results_task4_pets.csv \
  --out-dir experiments/artifacts_task4

自定义组合（可重复传多个 `--combo`）：

```zsh
source .venv/bin/activate
python -m experiments.task4_runner \
  --combo 'tag=my_combo;pre=gaussian_filter,normalizer_minmax_image;feat=hog;clf=svm_linear' \
  --out-csv experiments/results_custom.csv \
  --out-dir experiments/artifacts_custom
```

## 前端（Streamlit）

提供一个最小前端：数据集固定为 Pets，三个环节方法可选；**预处理支持叠加并可调整先后顺序**，点击 Run 生成同 schema CSV。

```zsh
cd /Users/hh/MacCodes/PatternRecognition
source .venv/bin/activate
streamlit run web/app.py
```
```

快速自测（使用 FakeData，不依赖 Pets 下载）：

```zsh
source .venv/bin/activate
python -m experiments.task4_smoke_test
```

# PatternRecognition
# PatternRecognition
