from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Row:
    dataset: str
    split_seed: int
    tag: str
    accuracy: float
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
                accuracy=_to_float(r.get("accuracy", 0.0)),
                confusion_path=str(r.get("confusion_path", "")),
            )
        )
    return rows


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


def _load_cm(path_str: str) -> np.ndarray:
    p = Path(path_str)
    if not p.exists():
        p = Path(".") / path_str
    if not p.exists():
        raise FileNotFoundError(str(p))
    cm = np.asarray(json.loads(p.read_text(encoding="utf-8")), dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"invalid confusion matrix shape: {cm.shape}")
    return cm


def _load_class_names(dataset: str) -> list[str] | None:
    if dataset != "oxford_pets":
        return None

    try:
        from data.torch_datasets import load_oxford_pets

        bundle = load_oxford_pets(root="data/oxford_pets", image_size=224, batch_size=4, num_workers=0, seed=0)
        if bundle.class_names:
            return list(bundle.class_names)
    except Exception:
        return None

    return None


def _topk_confusions(cm: np.ndarray, *, k: int) -> list[tuple[int, int, float, int]]:
    """返回 top-k 的 (true_i, pred_j, rate, count)，按 rate 降序。"""

    n = int(cm.shape[0])
    row_sum = cm.sum(axis=1, keepdims=True)
    rate = np.divide(cm, np.maximum(row_sum, 1e-12))

    triples: list[tuple[int, int, float, int]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            triples.append((i, j, float(rate[i, j]), int(cm[i, j])))

    triples.sort(key=lambda x: (x[2], x[3]), reverse=True)
    return triples[:k]


def _topk_pairs(cm: np.ndarray, *, k: int) -> list[tuple[int, int, float, int]]:
    """返回 top-k 的无序类别对 (a,b, pair_rate, pair_count)，按 pair_rate 降序。"""

    n = int(cm.shape[0])
    row_sum = cm.sum(axis=1, keepdims=True)
    rate = np.divide(cm, np.maximum(row_sum, 1e-12))

    pairs: list[tuple[int, int, float, int]] = []
    for a in range(n):
        for b in range(a + 1, n):
            pair_rate = float(rate[a, b] + rate[b, a])
            pair_count = int(cm[a, b] + cm[b, a])
            pairs.append((a, b, pair_rate, pair_count))

    pairs.sort(key=lambda x: (x[2], x[3]), reverse=True)
    return pairs[:k]


def _name(idx: int, names: list[str] | None) -> str:
    if names and 0 <= idx < len(names):
        return f"{names[idx]}"
    return f"class_{idx}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Extract Top-K most confused class pairs from the best model confusion matrix")
    p.add_argument(
        "--inputs",
        nargs="+",
        default=["experiments/sweep_all_leaderboard.csv", "experiments/transfer_more_epochs_raw.csv"],
        help="one or more CSV files (leaderboard/raw) to merge",
    )
    p.add_argument("--out", default="report/assets/top_confusions.md")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args(argv)

    rows = _merge_unique([_read_rows(Path(x)) for x in list(args.inputs)])
    if not rows:
        raise RuntimeError("no rows loaded")

    best = max(rows, key=lambda r: r.accuracy)
    if not best.confusion_path:
        raise RuntimeError("best row has no confusion_path")

    cm = _load_cm(best.confusion_path)
    names = _load_class_names(best.dataset)

    top_dir = _topk_confusions(cm, k=int(args.k))
    top_pair = _topk_pairs(cm, k=int(args.k))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# 最优模型 Top-K 易混淆类别统计\n\n")
    lines.append(f"最优模型 tag：{best.tag}（accuracy={best.accuracy:.4f}）\n\n")
    lines.append(
        "说明：混淆矩阵元素 $C_{ij}$ 表示真实类别 $i$ 被预测为 $j$ 的样本数。本文同时给出：\n"
        "- **方向性混淆**：$i\\to j$ 的行归一化比例 $C_{ij}/\\sum_j C_{ij}$（更能反映该类被错分到哪一类）。\n"
        "- **类别对混淆**：$(a,b)$ 的双向混淆比例之和（$a\\to b$ 与 $b\\to a$）。\n\n"
    )

    lines.append(f"## 方向性 Top-{int(args.k)}（按行归一化比例）\n\n")
    lines.append("|#|true|pred|比例(行归一化)|错分数|\n|---:|---|---|---:|---:|\n")
    for idx, (i, j, rate, cnt) in enumerate(top_dir, start=1):
        lines.append(f"|{idx}|{_name(i, names)}|{_name(j, names)}|{rate:.4f}|{cnt}|\n")
    lines.append("\n")

    lines.append(f"## 类别对 Top-{int(args.k)}（双向混淆之和）\n\n")
    lines.append("|#|类别A|类别B|双向比例和|双向错分数|\n|---:|---|---|---:|---:|\n")
    for idx, (a, b, pair_rate, pair_cnt) in enumerate(top_pair, start=1):
        lines.append(f"|{idx}|{_name(a, names)}|{_name(b, names)}|{pair_rate:.4f}|{pair_cnt}|\n")
    lines.append("\n")

    out.write_text("".join(lines), encoding="utf-8")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
