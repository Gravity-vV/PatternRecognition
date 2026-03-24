from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Row:
    dataset: str
    split_seed: int
    tag: str
    preprocess: str
    feature: str
    classifier: str
    accuracy: float
    f1: float
    train_time_sec: float
    inference_time_sec: float
    confusion_path: str


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _to_int(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def _read_rows(csv_path: Path) -> list[Row]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))

    rows: list[Row] = []
    for r in raw:
        rows.append(
            Row(
                dataset=str(r.get("dataset", "")),
                split_seed=_to_int(r.get("split_seed", 0)),
                tag=str(r.get("tag", "")),
                preprocess=str(r.get("preprocess", "")),
                feature=str(r.get("feature", "")),
                classifier=str(r.get("classifier", "")),
                accuracy=_to_float(r.get("accuracy", 0.0)),
                f1=_to_float(r.get("f1", 0.0)),
                train_time_sec=_to_float(r.get("train_time_sec", 0.0)),
                inference_time_sec=_to_float(r.get("inference_time_sec", 0.0)),
                confusion_path=str(r.get("confusion_path", "")),
            )
        )
    return rows


def _family(r: Row) -> str:
    if "convnext_tiny_embedding" in r.feature:
        return "embedding"
    if "TorchTransferClassifier" in r.classifier or "transfer" in r.tag:
        return "transfer"
    return "classic"


def _merge_unique(rows_list: list[list[Row]]) -> list[Row]:
    merged: list[Row] = []
    seen: set[tuple[str, int, str]] = set()
    for rows in rows_list:
        for r in rows:
            k = (r.dataset, int(r.split_seed), r.tag)
            if k in seen:
                continue
            seen.add(k)
            merged.append(r)
    return merged


def _safe_log10(x: float) -> float:
    return math.log10(max(1e-6, x))


def fig_scatter_accuracy_time(rows: list[Row], out_path: Path) -> None:
    # 每个 family 只取 Top-N，避免图太糊
    topn = 25
    groups: dict[str, list[Row]] = {"embedding": [], "transfer": [], "classic": []}
    for r in rows:
        groups[_family(r)].append(r)

    for k in list(groups.keys()):
        groups[k].sort(key=lambda x: x.accuracy, reverse=True)
        groups[k] = groups[k][:topn]

    colors = {"embedding": "tab:blue", "transfer": "tab:orange", "classic": "tab:gray"}
    labels = {"embedding": "Embedding", "transfer": "Transfer", "classic": "Classic"}

    plt.figure(figsize=(8.2, 4.8), dpi=160)
    for fam, pts in groups.items():
        if not pts:
            continue
        xs = [_safe_log10(p.train_time_sec) for p in pts]
        ys = [p.accuracy for p in pts]
        plt.scatter(xs, ys, s=40, alpha=0.85, c=colors[fam], label=labels[fam])

    plt.xlabel("log10(train_time_sec)")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs Training Time (Top per family)")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    plt.legend(frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _parse_embed_setting(preprocess: str) -> tuple[str, str]:
    # 返回 (augment, vec_norm)
    parts = [p.strip() for p in preprocess.split("+") if p.strip()]
    augment = "on" if "torch_augment_224" in parts else "off"
    vec_norm = "none"
    for p in parts:
        if p.startswith("normalizer_minmax_vec"):
            vec_norm = "minmax"
        elif p.startswith("normalizer_zscore_vec"):
            vec_norm = "zscore"
    return augment, vec_norm


def fig_embedding_ablation(rows: list[Row], out_path: Path) -> None:
    # 只看 embedding 且不接 PCA，便于解释
    emb = [r for r in rows if r.feature == "convnext_tiny_embedding" and _family(r) == "embedding"]
    if not emb:
        raise RuntimeError("no embedding rows found")

    # 取每个 (augment, vec_norm, classifier) 的 best accuracy
    best: dict[tuple[str, str, str], float] = {}
    for r in emb:
        aug, vnorm = _parse_embed_setting(r.preprocess)
        clf = r.classifier
        key = (aug, vnorm, clf)
        best[key] = max(best.get(key, 0.0), r.accuracy)

    # 固定顺序，便于读图
    aug_order = ["off", "on"]
    vnorm_order = ["none", "minmax", "zscore"]
    clf_order = ["logreg", "svm_linear", "svm_rbf", "random_forest"]

    series: list[tuple[str, list[float]]] = []
    xlabels: list[str] = []

    for aug in aug_order:
        for vnorm in vnorm_order:
            xlabels.append(f"aug={aug}\nvec={vnorm}")

    for clf in clf_order:
        ys: list[float] = []
        for aug in aug_order:
            for vnorm in vnorm_order:
                ys.append(best.get((aug, vnorm, clf), float("nan")))
        # 若该 clf 完全没有数据就跳过
        if all(np.isnan(y) for y in ys):
            continue
        series.append((clf, ys))

    x = np.arange(len(xlabels))
    width = 0.18

    plt.figure(figsize=(10.6, 4.8), dpi=160)

    # 更柔和的配色（避免过艳）
    cmap = plt.colormaps.get_cmap("Set2")
    palette = [cmap(i) for i in range(max(3, len(series)))]

    for i, (name, ys) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        plt.bar(
            x + offset,
            ys,
            width=width,
            label=name,
            color=palette[i % len(palette)],
            edgecolor=(0, 0, 0, 0.25),
            linewidth=0.6,
            alpha=0.90,
        )

    plt.ylim(0.85, 0.96)
    plt.ylabel("best accuracy")
    plt.title("Embedding ablation (best accuracy by augment / vector-normalizer / classifier)")
    plt.xticks(x, xlabels)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.legend(ncol=min(4, len(series)), frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def fig_confusion_best(rows: list[Row], out_path: Path) -> None:
    # 选全局最优（accuracy 最大）的那一行
    best = max(rows, key=lambda r: r.accuracy)
    if not best.confusion_path:
        raise RuntimeError("best row has no confusion_path")

    cm_path = Path(best.confusion_path)
    if not cm_path.exists():
        # 有些 CSV 里路径是相对项目根目录
        cm_path = Path(".") / best.confusion_path
    if not cm_path.exists():
        raise FileNotFoundError(str(cm_path))

    cm = np.asarray(json.loads(cm_path.read_text(encoding="utf-8")), dtype=np.float32)
    # 行归一化：每一类的 recall 分布更直观
    row_sum = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, np.maximum(row_sum, 1e-6))

    # Top-5 最大的非对角混淆（按行归一化比例）
    n = int(cm.shape[0])
    candidates: list[tuple[float, int, int]] = []  # (rate, i, j)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            candidates.append((float(cmn[i, j]), i, j))
    candidates.sort(reverse=True)
    top5 = [(i, j, r) for (r, i, j) in candidates[:5]]

    plt.figure(figsize=(7.2, 6.2), dpi=180)
    im = plt.imshow(cmn, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="row-normalized")

    plt.title(
        "Confusion matrix (row-normalized)\n"
        f"best={best.tag} | acc={best.accuracy:.4f}"
    )
    plt.xlabel("pred")
    plt.ylabel("true")

    # 37 类太密：不画每个 tick label，只保留刻度
    step = 5 if n > 20 else 1
    ticks = list(range(0, n, step))
    plt.xticks(ticks, [str(t) for t in ticks], fontsize=8)
    plt.yticks(ticks, [str(t) for t in ticks], fontsize=8)

    # 在图上编号标注 Top-5 混淆点（真实 i -> 预测 j）
    for idx, (i, j, r) in enumerate(top5, start=1):
        plt.text(
            j,
            i,
            str(idx),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="darkred",
            bbox=dict(boxstyle="round,pad=0.15", fc=(1, 1, 1, 0.65), ec=(0.5, 0, 0, 0.55), lw=0.8),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate 3 key figures (PNG) for the Chinese report")
    p.add_argument(
        "--inputs",
        nargs="+",
        default=["experiments/sweep_all_leaderboard.csv", "experiments/transfer_more_epochs_raw.csv"],
        help="one or more CSV files (leaderboard/raw) to merge",
    )
    p.add_argument("--out-dir", default="report/figures")
    args = p.parse_args(argv)

    rows_list = [_read_rows(Path(x)) for x in list(args.inputs)]
    rows = _merge_unique(rows_list)
    if not rows:
        raise RuntimeError("no rows loaded")

    out_dir = Path(args.out_dir)

    fig_scatter_accuracy_time(rows, out_dir / "fig1_acc_vs_train_time.png")
    fig_embedding_ablation(rows, out_dir / "fig2_embedding_ablation.png")
    fig_confusion_best(rows, out_dir / "fig3_confusion_best.png")

    print(f"wrote: {out_dir / 'fig1_acc_vs_train_time.png'}")
    print(f"wrote: {out_dir / 'fig2_embedding_ablation.png'}")
    print(f"wrote: {out_dir / 'fig3_confusion_best.png'}")


if __name__ == "__main__":
    main()
