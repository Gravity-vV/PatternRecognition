from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.task4_runner import ComboSpec, run_task4  # noqa: E402
from modules.registry import MethodInfo, get_registry  # noqa: E402


st.set_page_config(page_title="Pets 模式识别评测", layout="wide")


@dataclass
class MethodOption:
    name: str
    label: str


def _opts(methods: list[MethodInfo]) -> list[MethodOption]:
    return [MethodOption(m.name, f"{m.label}  (`{m.name}`)") for m in methods]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(row))
    return rows


def _coerce_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _build_tag(pre: list[str], feat: str, clf: str) -> str:
    p = "-".join(pre) if pre else "none"
    return f"pre[{p}]__feat[{feat}]__clf[{clf}]"[:120]


def _override_params(
    split_seed: int,
    *,
    features: list[str],
    classifier: str,
    pretrained: bool,
    epochs: int,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    params: dict[str, dict[str, Any]] = {}

    # 常见可复现参数
    if "pca" in features:
        params.setdefault("pca", {})["random_state"] = split_seed
    if classifier in {"random_forest", "svm_linear", "svm_rbf", "logreg"}:
        params.setdefault(classifier, {})["random_state"] = split_seed

    if "convnext_tiny_embedding" in features:
        params.setdefault("convnext_tiny_embedding", {})["seed"] = split_seed
        params.setdefault("convnext_tiny_embedding", {})["pretrained"] = pretrained

    if classifier == "convnext_tiny_transfer":
        params.setdefault("convnext_tiny_transfer", {})["seed"] = split_seed
        params.setdefault("convnext_tiny_transfer", {})["epochs"] = epochs
        params.setdefault("convnext_tiny_transfer", {})["batch_size"] = batch_size
        params.setdefault("convnext_tiny_transfer", {})["pretrained"] = pretrained

    return params


st.title("Pets 模式识别评测（可组合三环节）")
st.caption("数据集固定为 Oxford-IIIT Pets；图像预处理单选 + 向量预处理单选，便于与 sweep 一致对比。")

reg = get_registry()

pre_opts = _opts(reg.list_by_category("preprocessing"))
feat_opts = _opts(reg.list_by_category("feature"))
clf_opts = _opts(reg.list_by_category("classifier"))

# 与 --sweep 保持一致：特征阶段是“选 1 个选项”，该选项内部可能由多个 feature 串起来
FEATURE_OPTIONS: dict[str, list[str]] = {
    "hog": ["hog"],
    "hog+pca": ["hog", "pca"],
    "lbp": ["lbp"],
    "lbp+pca": ["lbp", "pca"],
    "color_hist": ["color_hist"],
    "color_hist+pca": ["color_hist", "pca"],
    "handcrafted_fusion": ["handcrafted_fusion"],
    "handcrafted_fusion+pca": ["handcrafted_fusion", "pca"],
    "convnext_tiny_embedding": ["convnext_tiny_embedding"],
    "convnext_tiny_embedding+pca": ["convnext_tiny_embedding", "pca"],
}

with st.sidebar:
    st.header("运行配置")
    split_seed = st.number_input("split_seed", min_value=0, max_value=10000, value=0, step=1)
    epochs = st.number_input("epochs（仅 convnext_tiny_transfer）", min_value=1, max_value=50, value=1, step=1)
    batch_size = st.number_input("batch_size", min_value=1, max_value=256, value=32, step=1)
    pretrained = st.checkbox("ConvNeXt 使用预训练权重（会缓存复用）", value=True)

    out_csv = st.text_input("输出 CSV", value="experiments/results_task4_pets.csv")
    out_dir = st.text_input("输出 artifacts 目录", value="experiments/artifacts_task4")

# --- UI 顺序：1) 预处理 -> 2) 特征 -> 3) 分类器 ---
# 预处理的可选项依赖于是否为 torch 管线；这里先从 session_state 读取“上一次选择”来决定展示。
_FEAT_KEY = "feat_choice"
_CLF_KEY = "clf_choice"

clf_labels = [o.label for o in clf_opts]
clf_map = {o.label: o.name for o in clf_opts}

feat_choice_preview = st.session_state.get(_FEAT_KEY, "hog")
features_preview = [] if feat_choice_preview == "(none)" else list(FEATURE_OPTIONS.get(feat_choice_preview, ["hog"]))

clf_choice_preview = st.session_state.get(_CLF_KEY, clf_labels[0] if clf_labels else "")
classifier_name_preview = clf_map.get(clf_choice_preview, clf_map[clf_labels[0]] if clf_labels else "")

is_torch_pipeline = classifier_name_preview == "convnext_tiny_transfer" or (
    features_preview and features_preview[0] == "convnext_tiny_embedding"
)

st.subheader("1) 数据预处理（单选）")

img_pre_all = reg.list_by_category("preprocessing")
numpy_img_pre = [m for m in img_pre_all if m.input_kind == "numpy_image" and m.output_kind == "numpy_image"]
torch_img_pre = [m for m in img_pre_all if m.input_kind == "torch_image" and m.output_kind == "torch_image"]
vec_pre = [m for m in img_pre_all if m.input_kind == "vector" and m.output_kind == "vector"]

if is_torch_pipeline:
    img_pre_labels = ["(default)"] + [f"{m.label}  (`{m.name}`)" for m in torch_img_pre]
    img_pre_map = {f"{m.label}  (`{m.name}`)": m.name for m in torch_img_pre}
    img_pre_choice = st.selectbox(
        "图像预处理（torch_image）",
        options=img_pre_labels,
        index=0,
        key="img_pre_choice_torch",
    )
    img_pre_name = "" if img_pre_choice == "(default)" else img_pre_map[img_pre_choice]
else:
    img_pre_labels = ["(none)"] + [f"{m.label}  (`{m.name}`)" for m in numpy_img_pre]
    img_pre_map = {f"{m.label}  (`{m.name}`)": m.name for m in numpy_img_pre}
    img_pre_choice = st.selectbox(
        "图像预处理（numpy_image）",
        options=img_pre_labels,
        index=0,
        key="img_pre_choice_numpy",
    )
    img_pre_name = "" if img_pre_choice == "(none)" else img_pre_map[img_pre_choice]

vec_pre_labels = ["(none)"] + [f"{m.label}  (`{m.name}`)" for m in vec_pre]
vec_pre_map = {f"{m.label}  (`{m.name}`)": m.name for m in vec_pre}
vec_pre_choice = st.selectbox("向量预处理（vector）", options=vec_pre_labels, index=0, key="vec_pre_choice")
vec_pre_name = "" if vec_pre_choice == "(none)" else vec_pre_map[vec_pre_choice]

preprocess = [n for n in [img_pre_name, vec_pre_name] if n]

st.subheader("2) 特征提取（选 1 个选项）")
feat_choice = st.selectbox(
    "选择特征选项",
    options=["(none)"] + list(FEATURE_OPTIONS.keys()),
    index=1,
    key=_FEAT_KEY,
)
features = [] if feat_choice == "(none)" else list(FEATURE_OPTIONS[feat_choice])

st.subheader("3) 分类器模型")
clf_choice = st.selectbox("选择分类器", options=clf_labels, index=0, key=_CLF_KEY)
classifier_name = clf_map[clf_choice]

# 组合约束提示（最小提示，不做复杂 UI）
clf_info = reg.get(classifier_name)

if classifier_name == "convnext_tiny_transfer":
    if features:
        st.warning("选择了 convnext_tiny_transfer 时，特征提取必须为 (none)（端到端训练）。")
else:
    if not features:
        st.warning("当前分类器需要向量输入，请选择一个特征提取方法。")

run = st.button("Run", type="primary")

if run:
    try:
        feats: list[str] = list(features)
        params = _override_params(
            int(split_seed),
            features=feats,
            classifier=classifier_name,
            pretrained=bool(pretrained),
            epochs=int(epochs),
            batch_size=int(batch_size),
        )

        pre = preprocess
        tag = _build_tag(pre, feat_choice, classifier_name)
        combo = ComboSpec(tag=tag, preprocess=pre, features=feats, classifier=classifier_name, params=params)

        prog = st.progress(0.0, text="准备开始")
        stage = st.empty()

        def progress_cb(msg: str, frac: float) -> None:
            prog.progress(max(0.0, min(1.0, float(frac))), text=msg)
            stage.write(msg)

        wrote = run_task4(
            split_seed=int(split_seed),
            epochs=int(epochs),
            batch_size=int(batch_size),
            num_workers=0,
            pretrained=bool(pretrained),
            out_csv=Path(out_csv),
            out_dir=Path(out_dir),
            combos=[combo],
            tags=None,
            progress=progress_cb,
        )

        st.success(f"已生成：`{wrote}`")
        rows = _read_csv_rows(Path(wrote))
        if rows:
            # 展示结果（单行）
            st.dataframe(rows, use_container_width=True)
            acc = _coerce_float(rows[0].get("accuracy"))
            if acc is not None:
                st.metric("Accuracy", f"{acc:.4f}")
        else:
            st.info("CSV 为空")

    except Exception as e:
        st.exception(e)
