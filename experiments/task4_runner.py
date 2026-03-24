from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from data.pets_numpy import load_oxford_pets_numpy
from data.torch_datasets import default_device, load_oxford_pets
from modules.base import BaseModule
from modules.registry import MethodInfo, get_registry
from modules.vision_models import TorchTransferClassifier
from modules.vision_preprocessing import build_vision_transforms
from utils.metrics import compute_classification_metrics
from utils.timing import Timer, timed


@dataclass(frozen=True)
class ComboSpec:
    tag: str
    preprocess: list[str]
    features: list[str]
    classifier: str
    params: dict[str, dict[str, Any]] | None = None  # name -> overrides


def _write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _save_confusion(cm: np.ndarray, out_dir: Path, *, dataset: str, split_seed: int, tag: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"confusion_{dataset}_{split_seed}_{tag}.json"
    p.write_text(json.dumps(cm.tolist(), ensure_ascii=False))
    return str(p)


def _describe_chain(names: list[str]) -> str:
    return "+".join(names) if names else "none"


def _instantiate(reg, name: str, overrides: dict[str, Any]) -> BaseModule:
    return reg.create(name, **overrides)


def _validate_combo(reg, combo: ComboSpec) -> tuple[str, str]:
    """返回 (start_kind, end_kind) 并在不合法时抛错。

    说明：预处理列表里允许同时存在 image 与 vector 两类。
    - image 预处理会在特征提取前应用
    - vector 预处理会在数据进入 vector 域之后应用（例如 embedding 或 hog/pixel_flatten 之后）
    """

    clf = reg.get(combo.classifier)

    # start_kind 规则：若 classifier/feature/preprocess 任一需要 torch_image，则走 torch 流水线；否则走 numpy。
    need_torch = clf.input_kind == "torch_image" or any(
        reg.get(n).input_kind == "torch_image" for n in (combo.preprocess + combo.features)
    )
    start_kind = "torch_image" if need_torch else "numpy_image"

    # 粗校验：
    # - torch 起点：features 允许为空（端到端）或以 convnext_tiny_embedding 开头
    # - numpy 起点：features 不能为空，且第一步必须能消费 numpy_image
    if start_kind == "torch_image":
        if clf.input_kind == "vector":
            # torch->vector->clf：必须有 embedding
            if not combo.features or reg.get(combo.features[0]).input_kind != "torch_image":
                raise ValueError("torch_image 组合若使用向量分类器，features 需以 convnext_tiny_embedding 开头")
        return start_kind, clf.output_kind  # type: ignore[return-value]

    # numpy 起点
    if not combo.features:
        raise ValueError("numpy_image 组合必须提供 features（例如 pixel_flatten/hog/pixel_flatten+pca）")
    if reg.get(combo.features[0]).input_kind != "numpy_image":
        raise ValueError("numpy_image 组合的第一个 feature 必须消费 numpy_image")
    return start_kind, clf.output_kind  # type: ignore[return-value]


def _apply_numpy_preprocess(
    mods: list[BaseModule],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Xt = X_train
    for m in mods:
        m.fit(Xt, y=y_train)
        Xt = m.transform(Xt)

    Xv = X_test
    for m in mods:
        Xv = m.transform(Xv)

    return Xt, Xv


def _apply_vector_features(
    mods: list[BaseModule],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Xt = X_train
    for m in mods:
        m.fit(Xt, y=y_train)
        Xt = m.transform(Xt)

    Xv = X_test
    for m in mods:
        Xv = m.transform(Xv)

    return Xt, Xv


def _extract_y_from_loader(loader: Any) -> np.ndarray:
    ys: list[np.ndarray] = []
    for _x, y in loader:
        ys.append(np.asarray(y, dtype=np.int64))
    return np.concatenate(ys, axis=0) if ys else np.asarray([], dtype=np.int64)


def _build_torch_transforms(preprocess_names: list[str], *, train: bool) -> Any:
    """根据 preprocessing 名称构建 torchvision transforms。

    约定：
    - train=True 时若包含 torch_augment_224 则启用增强；否则仅 resize。
    - imagenet_normalize 必须存在（否则 ConvNeXt 相关特征会偏）。
    """

    from modules.vision_preprocessing import ImageNetNormalizer, Resize, TorchAugmentor

    # 默认策略（最少配置也能跑）：resize + imagenet_normalize
    resize = Resize(size=224)
    normalizer = ImageNetNormalizer()
    augmentor = TorchAugmentor(size=224)

    if "resize_224" in preprocess_names:
        resize = Resize(size=224)
    if "imagenet_normalize" in preprocess_names:
        normalizer = ImageNetNormalizer()

    use_aug = train and ("torch_augment_224" in preprocess_names)
    return build_vision_transforms(resize=resize, normalizer=normalizer, augmentor=augmentor if use_aug else None, train=train)


def run_task4(
    *,
    split_seed: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    pretrained: bool = True,
    out_csv: Path,
    out_dir: Path,
    # 新接口：传入 combos；若不传则使用默认组合
    combos: Optional[list[ComboSpec]] = None,
    # 兼容旧接口：tags 过滤
    tags: Optional[set[str]] = None,
    progress: Optional[Callable[[str, float], None]] = None,
    on_loss: Optional[Callable[[int, int, float], None]] = None,
) -> Path:
    reg = get_registry()

    # numpy 数据缓存：按 image_size 缓存，避免 sweep 时每个 combo 都重复解码图片。
    numpy_dataset_cache: dict[int, Any] = {}

    # 复用/缓存（仅对本次 run_task4 调用生效）
    torch_bundle = None
    torch_y_true: np.ndarray | None = None
    embedding_cache: dict[tuple[int, bool, tuple[str, ...]], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    if combos is None:
        combos = [
            ComboSpec(
                tag="classic_pca_rf",
                preprocess=[],
                features=["pixel_flatten", "pca"],
                classifier="random_forest",
                params={
                    "pca": {"n_components": 128, "random_state": split_seed},
                    "random_forest": {"random_state": split_seed},
                },
            ),
            ComboSpec(
                tag="classic_hog_svm",
                preprocess=["gaussian_filter", "normalizer_minmax_image"],
                features=["hog"],
                classifier="svm_linear",
                params={
                    "gaussian_filter": {"sigma": 0.8},
                    "svm_linear": {"random_state": split_seed},
                },
            ),
            ComboSpec(
                tag="convnext_embed_logreg",
                preprocess=["resize_224", "imagenet_normalize"],
                features=["convnext_tiny_embedding"],
                classifier="logreg",
                params={
                    "convnext_tiny_embedding": {"seed": split_seed, "pretrained": pretrained},
                    "logreg": {"random_state": split_seed},
                },
            ),
            ComboSpec(
                tag="convnext_transfer_head",
                preprocess=["torch_augment_224", "imagenet_normalize"],
                features=[],
                classifier="convnext_tiny_transfer",
                params={
                    "convnext_tiny_transfer": {
                        "num_classes": 37,
                        "mode": "head",
                        "epochs": epochs,
                        "lr": 1e-3,
                        "batch_size": batch_size,
                        "seed": split_seed,
                        "device": str(default_device()),
                        "pretrained": pretrained,
                    }
                },
            ),
        ]

    if tags is not None:
        combos = [c for c in combos if c.tag in tags]

    if not combos:
        raise ValueError("no combos to run")

    out_dir = Path(out_dir)
    out_csv = Path(out_csv)

    rows: list[dict[str, Any]] = []

    for idx, combo in enumerate(combos):
        if progress:
            progress(f"准备：校验组合 | tag={combo.tag}", idx / max(1, len(combos)))

        start_kind, _end_kind = _validate_combo(reg, combo)
        overrides = combo.params or {}

        train_timer = Timer()
        infer_timer = Timer()

        # ========== numpy image 流水线 ==========
        if start_kind == "numpy_image":
            if progress:
                progress(f"加载数据（Pets numpy） | tag={combo.tag}", idx / max(1, len(combos)) + 0.01)
            pre_image = [n for n in combo.preprocess if reg.get(n).input_kind == "numpy_image"]
            pre_vec = [n for n in combo.preprocess if reg.get(n).input_kind == "vector"]

            # 若选择了 resize_numpy_xxx，则直接按对应尺寸加载数据，避免从 64 上采样造成信息损失。
            # 仍保留 resize_numpy_xxx 在预处理链里：ResizeNumpyImage 会在尺寸已匹配时 no-op，
            # 这样输出的 preprocess 描述与实际执行保持一致。
            desired_size = 64
            if "resize_numpy_224" in pre_image:
                desired_size = 224
            elif "resize_numpy_128" in pre_image:
                desired_size = 128

            if desired_size not in numpy_dataset_cache:
                numpy_dataset_cache[desired_size] = load_oxford_pets_numpy(
                    root="data/oxford_pets",
                    image_size=desired_size,
                    seed=split_seed,
                )
            ds = numpy_dataset_cache[desired_size]

            preprocess_mods = [_instantiate(reg, n, overrides.get(n, {})) for n in pre_image]
            vec_pre_mods = [_instantiate(reg, n, overrides.get(n, {})) for n in pre_vec]

            feature_mods = [_instantiate(reg, n, overrides.get(n, {})) for n in combo.features]
            clf = _instantiate(reg, combo.classifier, overrides.get(combo.classifier, {}))

            with timed(train_timer):
                Xtr = ds.X_train
                Xte = ds.X_test

                # 预处理（numpy_image -> numpy_image）
                Xtr, Xte = _apply_numpy_preprocess(preprocess_mods, Xtr, ds.y_train, Xte)

                # 特征（可能包含：pixel_flatten/hog/pca 等）
                Xt = Xtr
                Xv = Xte
                yt = ds.y_train
                in_vector = False
                for m, name in zip(feature_mods, combo.features):
                    mi = reg.get(name)
                    if mi.input_kind == "numpy_image" and mi.output_kind == "vector":
                        Xt = m.transform(Xt)
                        Xv = m.transform(Xv)
                        in_vector = True
                        # 一旦进入 vector 域，先应用 vector 预处理（只做一次）
                        if vec_pre_mods:
                            Xt, Xv = _apply_vector_features(vec_pre_mods, np.asarray(Xt), yt, np.asarray(Xv))
                            vec_pre_mods = []
                    elif mi.input_kind == "vector" and mi.output_kind == "vector":
                        if not in_vector:
                            raise ValueError(f"feature {name} 需要 vector 输入，但当前仍在 numpy_image 域")
                        m.fit(np.asarray(Xt), y=yt)
                        Xt = m.transform(np.asarray(Xt))
                        Xv = m.transform(np.asarray(Xv))
                    else:
                        raise ValueError(f"unsupported feature step: {name} ({mi.input_kind}->{mi.output_kind})")

                # classifier 需要 vector
                if hasattr(clf, "fit"):
                    clf.fit(np.asarray(Xt), y=yt)

            with timed(infer_timer):
                y_pred = clf.predict(np.asarray(Xv))

            metrics = compute_classification_metrics(ds.y_test, np.asarray(y_pred))
            cm_path = _save_confusion(metrics.confusion, out_dir, dataset=ds.name, split_seed=split_seed, tag=combo.tag)

            model_path = ""
            try:
                import joblib

                model_file = out_dir / f"model_{ds.name}_{split_seed}_{combo.tag}.joblib"
                joblib.dump({"preprocess": preprocess_mods, "features": feature_mods, "classifier": clf}, model_file)
                model_path = str(model_file)
            except Exception:
                model_path = ""

            rows.append(
                {
                    "dataset": ds.name,
                    "split_seed": split_seed,
                    "tag": combo.tag,
                    "preprocess": _describe_chain(combo.preprocess),
                    "feature": _describe_chain(combo.features),
                    "classifier": combo.classifier,
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "train_time_sec": train_timer.elapsed,
                    "inference_time_sec": infer_timer.elapsed,
                    "n_train": int(ds.y_train.shape[0]),
                    "n_test": int(ds.y_test.shape[0]),
                    "model_path": model_path,
                    "confusion_path": cm_path,
                }
            )
            continue

        # ========== torch image 流水线（ConvNeXt embedding 或 端到端 transfer） ==========
        if start_kind == "torch_image":
            device = default_device()
            if progress:
                progress(f"加载数据（Pets DataLoader） | tag={combo.tag}", idx / max(1, len(combos)) + 0.01)
            if torch_bundle is None:
                torch_bundle = load_oxford_pets(
                    root="data/oxford_pets",
                    image_size=224,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    seed=split_seed,
                )
            bundle = torch_bundle

            train_tf = _build_torch_transforms(combo.preprocess, train=True)
            test_tf = _build_torch_transforms(combo.preprocess, train=False)

            def _pair(tf):
                return lambda img, target: (tf(img), target)

            bundle.train_loader.dataset.transform = train_tf  # type: ignore[attr-defined]
            bundle.test_loader.dataset.transform = test_tf  # type: ignore[attr-defined]
            bundle.train_loader.dataset.transforms = _pair(train_tf)  # type: ignore[attr-defined]
            bundle.test_loader.dataset.transforms = _pair(test_tf)  # type: ignore[attr-defined]

            # 情况 A：端到端 torch 分类器（convnext_tiny_transfer）
            if combo.classifier == "convnext_tiny_transfer":
                params = overrides.get(combo.classifier, {})
                params = {
                    "num_classes": bundle.num_classes,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "seed": split_seed,
                    "device": str(device),
                    "pretrained": pretrained,
                    **params,
                }
                clf = reg.create(combo.classifier, **params)
                if not isinstance(clf, TorchTransferClassifier):
                    raise RuntimeError("convnext_tiny_transfer should create TorchTransferClassifier")

                y_true = _extract_y_from_loader(bundle.test_loader)

                with timed(train_timer):
                    clf.fit(
                        bundle.train_loader,
                        on_progress=(lambda s, f: progress(f"{s} | tag={combo.tag}", f) if progress else None),
                        on_loss=on_loss,
                    )

                with timed(infer_timer):
                    y_pred = clf.predict(bundle.test_loader)

                metrics = compute_classification_metrics(y_true, y_pred)
                cm_path = _save_confusion(metrics.confusion, out_dir, dataset=bundle.name, split_seed=split_seed, tag=combo.tag)

                model_path = ""
                try:
                    model_file = out_dir / f"model_{bundle.name}_{split_seed}_{combo.tag}.pt"
                    model_path = clf.save(model_file)
                except Exception:
                    model_path = ""

                rows.append(
                    {
                        "dataset": bundle.name,
                        "split_seed": split_seed,
                        "tag": combo.tag,
                        "preprocess": _describe_chain(combo.preprocess),
                        "feature": _describe_chain(combo.features) if combo.features else "none",
                        "classifier": clf.label() if hasattr(clf, "label") else combo.classifier,
                        "accuracy": metrics.accuracy,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1": metrics.f1,
                        "train_time_sec": train_timer.elapsed,
                        "inference_time_sec": infer_timer.elapsed,
                        "n_train": int(len(bundle.train_loader.dataset)),
                        "n_test": int(len(bundle.test_loader.dataset)),
                        "model_path": model_path,
                        "confusion_path": cm_path,
                    }
                )
                continue

            # 情况 B：embedding 特征 -> 向量分类器
            pre_vec = [n for n in combo.preprocess if reg.get(n).input_kind == "vector"]
            vec_pre_mods = [_instantiate(reg, n, overrides.get(n, {})) for n in pre_vec]

            if not combo.features or combo.features[0] != "convnext_tiny_embedding":
                raise ValueError("torch_image -> vector 组合需要 features 以 convnext_tiny_embedding 开头")

            feat = _instantiate(reg, combo.features[0], overrides.get(combo.features[0], {}))

            # 抽 embedding（带缓存）。缓存 key 只与 torch_image 预处理有关；vector 预处理/分类器不影响 embedding。
            from torch.utils.data import DataLoader

            if torch_y_true is None:
                torch_y_true = _extract_y_from_loader(bundle.test_loader)

            torch_pre = tuple([n for n in combo.preprocess if reg.get(n).input_kind == "torch_image"])
            cache_key = (int(split_seed), bool(pretrained), torch_pre)
            cached = embedding_cache.get(cache_key)
            if cached is None:
                # 重要：train_loader 通常是 shuffle=True。
                # 若先单独遍历 loader 抽 y，再遍历一次抽 X，会导致两次遍历顺序不一致，从而 X/y 错位。
                train_loader = DataLoader(
                    bundle.train_loader.dataset,
                    batch_size=bundle.train_loader.batch_size,
                    shuffle=False,
                    num_workers=bundle.train_loader.num_workers,
                    pin_memory=getattr(bundle.train_loader, "pin_memory", False),
                )
                y_train = _extract_y_from_loader(train_loader)

                with timed(train_timer):
                    Xtr = feat.transform(train_loader)
                with timed(infer_timer):
                    Xte = feat.transform(bundle.test_loader)

                embedding_cache[cache_key] = (np.asarray(Xtr), np.asarray(Xte), np.asarray(y_train))
            else:
                Xtr, Xte, y_train = cached

            y_true = torch_y_true

            # embedding 后续的 vector 特征（如 pca）
            vec_feature_mods: list[BaseModule] = []
            for extra in combo.features[1:]:
                mi = reg.get(extra)
                if mi.input_kind != "vector":
                    raise ValueError(f"embedding 后续 features 仅支持 vector 类型，收到：{extra}")
                vec_feature_mods.append(_instantiate(reg, extra, overrides.get(extra, {})))

            # vector 域：先做 vector preprocess，再做 vector features
            Xt, Xv = _apply_vector_features(vec_pre_mods + vec_feature_mods, Xtr, y_train, Xte)

            clf = _instantiate(reg, combo.classifier, overrides.get(combo.classifier, {}))
            with timed(train_timer):
                clf.fit(Xt, y=y_train)
            with timed(infer_timer):
                y_pred = clf.predict(Xv)

            metrics = compute_classification_metrics(y_true, np.asarray(y_pred))
            cm_path = _save_confusion(metrics.confusion, out_dir, dataset=bundle.name, split_seed=split_seed, tag=combo.tag)

            model_path = ""
            try:
                import joblib

                model_file = out_dir / f"model_{bundle.name}_{split_seed}_{combo.tag}.joblib"
                joblib.dump({"preprocess": combo.preprocess, "features": combo.features, "classifier": combo.classifier, "clf": clf}, model_file)
                model_path = str(model_file)
            except Exception:
                model_path = ""

            rows.append(
                {
                    "dataset": bundle.name,
                    "split_seed": split_seed,
                    "tag": combo.tag,
                    "preprocess": _describe_chain(combo.preprocess),
                    "feature": _describe_chain(combo.features),
                    "classifier": combo.classifier,
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "train_time_sec": train_timer.elapsed,
                    "inference_time_sec": infer_timer.elapsed,
                    "n_train": int(len(bundle.train_loader.dataset)),
                    "n_test": int(len(bundle.test_loader.dataset)),
                    "model_path": model_path,
                    "confusion_path": cm_path,
                }
            )
            continue

        raise ValueError(f"unsupported start_kind: {start_kind}")

    if progress:
        progress("写入结果 CSV", 0.97)
    _write_csv(rows, out_csv)
    if progress:
        progress("完成", 1.0)
    return out_csv


def _parse_combo(s: str) -> ComboSpec:
    """解析 --combo 字符串。

    格式：
      tag=xxx;pre=a,b;feat=f1,f2;clf=c
    其中 pre/feat 可省略或为空。
    """

    parts = [p.strip() for p in s.split(";") if p.strip()]
    kv: dict[str, str] = {}
    for p in parts:
        if "=" not in p:
            raise ValueError(f"invalid combo part: {p}")
        k, v = p.split("=", 1)
        kv[k.strip()] = v.strip()

    tag = kv.get("tag", "combo")
    pre = [x.strip() for x in kv.get("pre", "").split(",") if x.strip()]
    feat = [x.strip() for x in kv.get("feat", "").split(",") if x.strip()]
    clf = kv.get("clf", "").strip()
    if not clf:
        raise ValueError("combo missing clf")
    return ComboSpec(tag=tag, preprocess=pre, features=feat, classifier=clf, params=None)


def _sanitize_tag_piece(s: str) -> str:
    return s.replace("+", "_").replace("/", "_").replace(" ", "_")


def _build_sweep_combos(
    *,
    split_seed: int,
    epochs: int,
    batch_size: int,
    pretrained: bool,
    include_numpy: bool,
    include_torch: bool,
    sweep_hyper: bool = False,
) -> list[ComboSpec]:
    reg = get_registry()

    vec_classifiers = ["random_forest", "svm_linear", "svm_rbf", "logreg"]

    # “特征提取”阶段：用户选 1 个选项（内部可能由多个 feature 串起来，例如 +pca）
    numpy_feature_options: dict[str, list[str]] = {
        "hog": ["hog"],
        "hog_pca": ["hog", "pca"],
        "lbp": ["lbp"],
        "lbp_pca": ["lbp", "pca"],
        "color_hist": ["color_hist"],
        "color_hist_pca": ["color_hist", "pca"],
        "handcrafted_fusion": ["handcrafted_fusion"],
        "handcrafted_fusion_pca": ["handcrafted_fusion", "pca"],
    }
    torch_feature_options: dict[str, list[str]] = {
        "convnext_embed": ["convnext_tiny_embedding"],
        "convnext_embed_pca": ["convnext_tiny_embedding", "pca"],
    }

    # “预处理”阶段：图像预处理与向量预处理分开，各自只选 1 个（允许 none）
    numpy_image_pre_options: list[list[str]] = [[]]
    torch_image_pre_options: list[list[str]] = [[]]
    vector_pre_options: list[list[str]] = [[]]

    for mi in reg.list_by_category("preprocessing"):
        if mi.input_kind == "numpy_image" and mi.output_kind == "numpy_image":
            numpy_image_pre_options.append([mi.name])
        elif mi.input_kind == "torch_image" and mi.output_kind == "torch_image":
            torch_image_pre_options.append([mi.name])
        elif mi.input_kind == "vector" and mi.output_kind == "vector":
            vector_pre_options.append([mi.name])

    def _merge_overrides(base: dict[str, dict[str, Any]], extra: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {k: dict(v) for k, v in base.items()}
        for name, params in extra.items():
            merged.setdefault(name, {})
            merged[name].update(params)
        return merged

    def _clf_variants(clf: str) -> list[tuple[str, dict[str, dict[str, Any]]]]:
        """返回 [(hp_tag, overrides)]。

        overrides 结构：module_name -> {param: value}
        """

        if not sweep_hyper:
            return [("default", {})]

        if clf == "svm_linear":
            variants: list[tuple[str, dict[str, dict[str, Any]]]] = []
            for C in (0.1, 1.0, 10.0):
                tag = f"svmC{str(C).replace('.', 'p')}"
                variants.append((tag, {"svm_linear": {"C": float(C)}}))
            return variants

        if clf == "svm_rbf":
            variants = []
            C_grid = (0.1, 1.0, 10.0)
            gamma_grid: tuple[str | float, ...] = ("scale", 0.01, 0.1)
            for C in C_grid:
                for gamma in gamma_grid:
                    gamma_tag = (
                        str(gamma).replace(".", "p").replace("-", "m")
                        if not isinstance(gamma, str)
                        else str(gamma)
                    )
                    tag = f"rbfC{str(C).replace('.', 'p')}_g{gamma_tag}"
                    variants.append((tag, {"svm_rbf": {"C": float(C), "gamma": gamma}}))
            return variants

        if clf == "random_forest":
            variants = []
            for n_estimators in (200, 500):
                for max_depth in (None, 20):
                    md_tag = "none" if max_depth is None else str(int(max_depth))
                    tag = f"rfN{int(n_estimators)}_d{md_tag}"
                    variants.append(
                        (
                            tag,
                            {"random_forest": {"n_estimators": int(n_estimators), "max_depth": max_depth}},
                        )
                    )
            return variants

        # logreg 或其它分类器：不做超参 sweep
        return [("default", {})]

    def _common_overrides(
        pre: list[str],
        feats: list[str],
        clf: str,
        *,
        extra_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        overrides: dict[str, dict[str, Any]] = {}
        if "pca" in feats:
            overrides.setdefault("pca", {})
            overrides["pca"].update({"random_state": split_seed})

        if clf in {"random_forest", "svm_linear", "svm_rbf", "logreg"}:
            overrides.setdefault(clf, {})
            overrides[clf].update({"random_state": split_seed})

        if "convnext_tiny_embedding" in feats:
            overrides.setdefault("convnext_tiny_embedding", {})
            overrides["convnext_tiny_embedding"].update({"seed": split_seed, "pretrained": pretrained})

        if clf == "convnext_tiny_transfer":
            overrides.setdefault("convnext_tiny_transfer", {})
            overrides["convnext_tiny_transfer"].update(
                {
                    "mode": "head",
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "seed": split_seed,
                    "device": str(default_device()),
                    "pretrained": pretrained,
                }
            )

        # 预处理默认参数一般足够，不强行覆盖
        _ = pre
        if extra_overrides:
            return _merge_overrides(overrides, extra_overrides)
        return overrides

    combos: list[ComboSpec] = []

    # Numpy：image -> (feature options) -> vector clf
    if include_numpy:
        for pre_img in numpy_image_pre_options:
            pre_img_tag = _sanitize_tag_piece(pre_img[0]) if pre_img else "none"
            for pre_vec in vector_pre_options:
                pre_vec_tag = _sanitize_tag_piece(pre_vec[0]) if pre_vec else "none"
                pre = pre_img + pre_vec
                for feat_tag, feats in numpy_feature_options.items():
                    for clf in vec_classifiers:
                        for hp_tag, hp_over in _clf_variants(clf):
                            tag = f"sweep_numpy__img-{pre_img_tag}__vec-{pre_vec_tag}__feat-{feat_tag}__clf-{clf}"
                            if sweep_hyper:
                                tag += f"__hp-{_sanitize_tag_piece(hp_tag)}"
                            combos.append(
                                ComboSpec(
                                    tag=tag,
                                    preprocess=pre,
                                    features=feats,
                                    classifier=clf,
                                    params=_common_overrides(pre, feats, clf, extra_overrides=hp_over),
                                )
                            )

    # Torch：torch image -> embedding -> vector clf
    # 注意：_build_torch_transforms 的默认已包含 resize_224+imagenet_normalize，
    # 所以 none/resize_224/imagenet_normalize 三者等价。只保留 none 和 torch_augment_224。
    if include_torch:
        torch_img_pre_dedup = [p for p in torch_image_pre_options if p and p[0] == "torch_augment_224"]
        torch_img_pre_dedup.insert(0, [])  # 插入 none（空）作为默认

        for pre_img in torch_img_pre_dedup:
            pre_img_tag = _sanitize_tag_piece(pre_img[0]) if pre_img else "none"
            for pre_vec in vector_pre_options:
                pre_vec_tag = _sanitize_tag_piece(pre_vec[0]) if pre_vec else "none"
                pre = pre_img + pre_vec
                for feat_tag, feats in torch_feature_options.items():
                    for clf in vec_classifiers:
                        for hp_tag, hp_over in _clf_variants(clf):
                            tag = f"sweep_torch_embed__img-{pre_img_tag}__vec-{pre_vec_tag}__feat-{feat_tag}__clf-{clf}"
                            if sweep_hyper:
                                tag += f"__hp-{_sanitize_tag_piece(hp_tag)}"
                            combos.append(
                                ComboSpec(
                                    tag=tag,
                                    preprocess=pre,
                                    features=feats,
                                    classifier=clf,
                                    params=_common_overrides(pre, feats, clf, extra_overrides=hp_over),
                                )
                            )

        # Torch：端到端 transfer（feature 必须为空）
        # 同样去重 torch image 预处理
        for pre_img in torch_img_pre_dedup:
            pre_img_tag = _sanitize_tag_piece(pre_img[0]) if pre_img else "none"
            clf = "convnext_tiny_transfer"
            tag = f"sweep_torch_transfer__img-{pre_img_tag}__feat-none__clf-{clf}"
            combos.append(
                ComboSpec(
                    tag=tag,
                    preprocess=pre_img,
                    features=[],
                    classifier=clf,
                    params=_common_overrides(pre_img, [], clf),
                )
            )

    # 最后再用 runner 的校验器过滤掉不合法的组合（防御式）
    valid: list[ComboSpec] = []
    for c in combos:
        try:
            _validate_combo(reg, c)
            valid.append(c)
        except Exception:
            continue
    return valid


def _rank_csv(in_csv: Path, *, sort_by: str = "accuracy") -> list[dict[str, Any]]:
    with in_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    def _key(r: dict[str, Any]) -> float:
        try:
            return float(r.get(sort_by, 0.0))
        except Exception:
            return 0.0

    rows.sort(key=_key, reverse=True)
    return rows


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Pets runner: composable preprocess/feature/classifier -> same schema CSV")
    p.add_argument("--dataset", default="oxford_pets", choices=["oxford_pets"])
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use pretrained ConvNeXt weights (cached under .cache/torch unless TORCH_HOME is set)",
    )
    p.add_argument("--out-csv", default="experiments/results_task4_pets.csv")
    p.add_argument("--out-dir", default="experiments/artifacts_task4")
    p.add_argument(
        "--tags",
        default="",
        help="comma-separated preset tags to run; empty means run all presets",
    )
    p.add_argument(
        "--combo",
        action="append",
        default=[],
        help="custom combo. format: tag=xxx;pre=a,b;feat=f1,f2;clf=c. can be repeated",
    )
    p.add_argument(
        "--combo-json",
        default="",
        help="path to a JSON file describing a combo (or a list of combos) including params overrides",
    )

    p.add_argument(
        "--sweep",
        action="store_true",
        help="run a sweep over combinations (single preprocess + one feature option + classifier) and rank by accuracy",
    )
    p.add_argument(
        "--include-numpy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include numpy pipeline combinations in --sweep",
    )
    p.add_argument(
        "--include-torch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include torch pipeline combinations in --sweep",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only run first N combos (0 means no limit). Useful for quick smoke tests",
    )
    p.add_argument(
        "--rank-metric",
        default="accuracy",
        choices=["accuracy", "precision", "recall", "f1"],
        help="metric used to rank results (only for printing/writing leaderboard)",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=20,
        help="print top-k rows after running (only when --sweep)",
    )
    p.add_argument(
        "--leaderboard-csv",
        default="",
        help="optional path to write a sorted leaderboard CSV (only when --sweep)",
    )

    p.add_argument(
        "--sweep-hyper",
        action="store_true",
        help="when used with --sweep, also enumerate hyperparameter grids for traditional classifiers (SVM/RF)",
    )

    args = p.parse_args(argv)

    tags = None
    if args.tags.strip():
        tags = {t.strip() for t in args.tags.split(",") if t.strip()}

    combos = None
    if args.sweep:
        combos = _build_sweep_combos(
            split_seed=args.split_seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            pretrained=bool(args.pretrained),
            include_numpy=bool(args.include_numpy),
            include_torch=bool(args.include_torch),
            sweep_hyper=bool(args.sweep_hyper),
        )
        if args.limit and args.limit > 0:
            combos = combos[: int(args.limit)]
    elif str(args.combo_json).strip():
        raw = Path(str(args.combo_json)).read_text(encoding="utf-8")
        obj = json.loads(raw)

        def _from_dict(d: dict[str, Any]) -> ComboSpec:
            return ComboSpec(
                tag=str(d.get("tag", "combo")),
                preprocess=list(d.get("preprocess", [])),
                features=list(d.get("features", [])),
                classifier=str(d.get("classifier", "")),
                params=d.get("params") or None,
            )

        if isinstance(obj, list):
            combos = [_from_dict(x) for x in obj]
        elif isinstance(obj, dict):
            combos = [_from_dict(obj)]
        else:
            raise ValueError("--combo-json must be a dict or a list of dicts")

        # 简单校验
        for c in combos:
            if not c.classifier:
                raise ValueError("combo-json missing classifier")
    elif args.combo:
        combos = [_parse_combo(s) for s in args.combo]

    progress_cb: Optional[Callable[[str, float], None]] = None
    if args.sweep:
        last_bucket = -1
        last_msg = ""

        def _progress(msg: str, frac: float) -> None:
            nonlocal last_bucket, last_msg
            bucket = int(max(0.0, min(1.0, float(frac))) * 100)
            if bucket != last_bucket or msg != last_msg:
                print(f"[{bucket:3d}%] {msg}")
                last_bucket = bucket
                last_msg = msg

        progress_cb = _progress

    out_csv = run_task4(
        split_seed=args.split_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrained=bool(args.pretrained),
        out_csv=Path(args.out_csv),
        out_dir=Path(args.out_dir),
        combos=combos,
        tags=tags,
        progress=progress_cb,
    )

    if args.sweep:
        ranked = _rank_csv(out_csv, sort_by=str(args.rank_metric))
        print(f"wrote: {out_csv}")
        print(f"leaderboard (sorted by {args.rank_metric}, top {int(args.topk)}):")
        for i, r in enumerate(ranked[: int(args.topk)], start=1):
            print(
                f"#{i:02d} acc={float(r['accuracy']):.4f} f1={float(r['f1']):.4f} | tag={r['tag']} | pre={r['preprocess']} | feat={r['feature']} | clf={r['classifier']}"
            )

        if str(args.leaderboard_csv).strip():
            lb = Path(str(args.leaderboard_csv))
            _write_csv(ranked, lb)
            print(f"wrote leaderboard: {lb}")
    else:
        print(f"wrote: {out_csv}")


if __name__ == "__main__":
    main()
