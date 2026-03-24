from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from data.pets_numpy import load_oxford_pets_numpy
from modules.classifiers import RandomForestClassifier, SVMClassifier
from modules.feature_extraction import HandcraftedFusionExtractor
from modules.preprocessing import Normalizer
from utils.metrics import compute_classification_metrics
from utils.timing import Timer, timed


def _write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _chunked_transform(extractor: HandcraftedFusionExtractor, X: np.ndarray, *, chunk: int, label: str) -> np.ndarray:
    feats: list[np.ndarray] = []
    n = int(X.shape[0])
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        feats.append(extractor.transform(X[i:j]))
        if (i // chunk) % 5 == 0:
            print(f"{label}: {j}/{n}")
    return np.concatenate(feats, axis=0)


def main() -> None:
    # 经验过滤（高收益、低浪费）：
    # - Pets 上传统方法大概率需要更高分辨率 + 向量 zscore。
    # - 特征先固定为 HOG+LBP+ColorHist 融合，避免每个超参组合重复算特征。
    # - 分类器优先 RBF SVM，小网格；Linear SVM/RF 作为对照。

    split_seed = 0
    image_size = 224
    chunk = 64

    print(f"loading pets numpy (image_size={image_size}) ...")
    ds = load_oxford_pets_numpy(root="data/oxford_pets", image_size=image_size, seed=split_seed)
    print(f"loaded: n_train={ds.y_train.shape[0]} n_test={ds.y_test.shape[0]}")

    print("extracting handcrafted_fusion features (once) ...")
    fusion = HandcraftedFusionExtractor()
    Xtr = _chunked_transform(fusion, ds.X_train, chunk=chunk, label="feat_train")
    Xte = _chunked_transform(fusion, ds.X_test, chunk=chunk, label="feat_test")
    print(f"feature dims: {Xtr.shape[1]}")

    print("applying vector zscore ...")
    norm = Normalizer(method="zscore", axis=0)
    norm.fit(Xtr)
    Xtr = norm.transform(Xtr)
    Xte = norm.transform(Xte)

    # 构造“可复现 combo json”（供 task4_runner --combo-json 复跑)
    pre_names = ["resize_numpy_224", "normalizer_zscore_vec"]
    feat_names = ["handcrafted_fusion"]

    def _combo_json(tag: str, classifier: str, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "tag": tag,
            "preprocess": pre_names,
            "features": feat_names,
            "classifier": classifier,
            "params": {classifier: params},
        }

    # 搜索空间（过滤后）
    rbf_grid = [(C, gamma) for C in (1.0, 10.0, 30.0) for gamma in ("scale", 0.01, 0.03)]
    lin_grid = [0.1, 1.0, 10.0]
    rf_grid = [(n, d) for n in (500, 1000) for d in (None, 30)]

    rows: list[dict[str, Any]] = []
    combos_json: list[dict[str, Any]] = []

    def _run_one(tag: str, clf_name: str, clf_obj, params: dict[str, Any]) -> None:
        train_timer = Timer()
        infer_timer = Timer()
        with timed(train_timer):
            clf_obj.fit(Xtr, y=ds.y_train)
        with timed(infer_timer):
            y_pred = clf_obj.predict(Xte)
        metrics = compute_classification_metrics(ds.y_test, y_pred)
        rows.append(
            {
                "dataset": ds.name,
                "split_seed": split_seed,
                "tag": tag,
                "preprocess": "+".join(pre_names),
                "feature": "+".join(feat_names),
                "classifier": clf_name,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "train_time_sec": train_timer.elapsed,
                "inference_time_sec": infer_timer.elapsed,
                "n_train": int(ds.y_train.shape[0]),
                "n_test": int(ds.y_test.shape[0]),
                "model_path": "",
                "confusion_path": "",
            }
        )
        combos_json.append(_combo_json(tag, clf_name, params))
        print(f"done: acc={metrics.accuracy:.4f} f1={metrics.f1:.4f} | {tag}")

    total = len(rbf_grid) + len(lin_grid) + len(rf_grid)
    done = 0

    # RBF SVM
    for C, gamma in rbf_grid:
        done += 1
        tag = (
            "trad_best_try__img-resize224__vec-zscore__feat-fusion__clf-svm_rbf"
            f"__C{str(C).replace('.', 'p')}__g{str(gamma).replace('.', 'p')}"
        )
        print(f"[{done}/{total}] train svm_rbf C={C} gamma={gamma} ...")
        clf = SVMClassifier(kernel="rbf", C=float(C), gamma=gamma, random_state=split_seed)
        _run_one(tag, "svm_rbf", clf, {"C": float(C), "gamma": gamma, "random_state": split_seed})

    # Linear SVM
    for C in lin_grid:
        done += 1
        tag = "trad_best_try__img-resize224__vec-zscore__feat-fusion__clf-svm_linear" f"__C{str(C).replace('.', 'p')}"
        print(f"[{done}/{total}] train svm_linear C={C} ...")
        clf = SVMClassifier(kernel="linear", C=float(C), gamma="scale", random_state=split_seed)
        _run_one(tag, "svm_linear", clf, {"C": float(C), "random_state": split_seed})

    # RandomForest
    for n_estimators, max_depth in rf_grid:
        done += 1
        md = "none" if max_depth is None else str(int(max_depth))
        tag = f"trad_best_try__img-resize224__vec-zscore__feat-fusion__clf-rf__N{int(n_estimators)}__d{md}"
        print(f"[{done}/{total}] train rf n_estimators={n_estimators} max_depth={max_depth} ...")
        clf = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            random_state=split_seed,
            n_jobs=-1,
        )
        _run_one(
            tag,
            "random_forest",
            clf,
            {"n_estimators": int(n_estimators), "max_depth": max_depth, "random_state": split_seed},
        )

    out_csv = Path("experiments/trad_best_targeted_raw.csv")
    _write_csv(rows, out_csv)
    print("wrote results:", out_csv)

    combos_path = Path("experiments/trad_best_targeted_combos.json")
    combos_path.write_text(json.dumps(combos_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote combos json:", combos_path)

    rows.sort(key=lambda r: float(r["accuracy"]), reverse=True)
    print("top 10:")
    for i, r in enumerate(rows[:10], start=1):
        print(
            f"#{i:02d} acc={float(r['accuracy']):.4f} f1={float(r['f1']):.4f} "
            f"clf={r['classifier']} tag={r['tag']}"
        )

    if rows:
        best_tag = rows[0]["tag"]
        best_combo = next((c for c in combos_json if c.get("tag") == best_tag), None)
        if best_combo is not None:
            best_json = Path("experiments/trad_best_targeted_best.json")
            best_json.write_text(json.dumps(best_combo, ensure_ascii=False, indent=2), encoding="utf-8")
            print("best combo json:", best_json)
            print("reproduce with runner:")
            print(
                "  python -m experiments.task4_runner --combo-json experiments/trad_best_targeted_best.json "
                "--out-csv experiments/trad_best_repro.csv"
            )


if __name__ == "__main__":
    main()
