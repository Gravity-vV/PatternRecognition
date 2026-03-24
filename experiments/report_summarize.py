from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
    precision: float
    recall: float
    train_time_sec: float
    inference_time_sec: float
    n_train: int
    n_test: int
    model_path: str
    confusion_path: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def _parse_rows(rows: Iterable[dict[str, str]]) -> list[Row]:
    parsed: list[Row] = []
    for r in rows:
        parsed.append(
            Row(
                dataset=str(r.get("dataset", "")),
                split_seed=_to_int(r.get("split_seed", 0)),
                tag=str(r.get("tag", "")),
                preprocess=str(r.get("preprocess", "")),
                feature=str(r.get("feature", "")),
                classifier=str(r.get("classifier", "")),
                accuracy=_to_float(r.get("accuracy", 0.0)),
                f1=_to_float(r.get("f1", 0.0)),
                precision=_to_float(r.get("precision", 0.0)),
                recall=_to_float(r.get("recall", 0.0)),
                train_time_sec=_to_float(r.get("train_time_sec", 0.0)),
                inference_time_sec=_to_float(r.get("inference_time_sec", 0.0)),
                n_train=_to_int(r.get("n_train", 0)),
                n_test=_to_int(r.get("n_test", 0)),
                model_path=str(r.get("model_path", "")),
                confusion_path=str(r.get("confusion_path", "")),
            )
        )
    return parsed


def _pipeline_family(row: Row) -> str:
    feat = row.feature
    clf = row.classifier

    if "convnext_tiny_embedding" in feat:
        return "torch_embedding"
    if "TorchTransferClassifier" in clf or "convnext_tiny_transfer" in row.tag:
        return "torch_transfer"
    return "classic_numpy"


def _md_table(rows: list[Row], *, title: str, max_rows: int = 10) -> str:
    header = (
        f"### {title}\n\n"
        "|#|acc|f1|train(s)|infer(s)|preprocess|feature|classifier|tag|\n"
        "|---:|---:|---:|---:|---:|---|---|---|---|\n"
    )
    lines = [header]
    for i, r in enumerate(rows[:max_rows], start=1):
        lines.append(
            "|{i}|{acc:.4f}|{f1:.4f}|{tr:.2f}|{inf:.3f}|{pre}|{feat}|{clf}|{tag}|\n".format(
                i=i,
                acc=r.accuracy,
                f1=r.f1,
                tr=r.train_time_sec,
                inf=r.inference_time_sec,
                pre=r.preprocess,
                feat=r.feature,
                clf=r.classifier,
                tag=r.tag,
            )
        )
    lines.append("\n")
    return "".join(lines)


def _top_by_family(rows: list[Row]) -> dict[str, list[Row]]:
    buckets: dict[str, list[Row]] = {}
    for r in rows:
        buckets.setdefault(_pipeline_family(r), []).append(r)

    for k in list(buckets.keys()):
        buckets[k].sort(key=lambda x: x.accuracy, reverse=True)

    return buckets


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Summarize sweep leaderboard into report-friendly markdown tables")
    p.add_argument(
        "--inputs",
        nargs="+",
        default=["experiments/sweep_all_leaderboard.csv"],
        help="one or more CSV paths (leaderboard or raw). rows are merged before ranking",
    )
    p.add_argument("--out", default="report/assets/auto_tables.md")
    p.add_argument("--topk", type=int, default=10)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    merged: list[Row] = []
    seen_key: set[tuple[str, int, str]] = set()
    for pth in list(args.inputs):
        src = Path(str(pth))
        if not src.exists():
            raise FileNotFoundError(str(src))
        rows_raw = _read_csv(src)
        for r in _parse_rows(rows_raw):
            k = (r.dataset, int(r.split_seed), r.tag)
            if k in seen_key:
                continue
            seen_key.add(k)
            merged.append(r)

    rows = merged
    rows.sort(key=lambda x: x.accuracy, reverse=True)

    buckets = _top_by_family(rows)

    parts: list[str] = []
    parts.append("# 自动汇总表（由 experiments/report_summarize.py 生成）\n\n")
    parts.append("来源 CSV：\n\n")
    for pth in list(args.inputs):
        parts.append(f"- {pth}\n")
    parts.append("\n")

    parts.append(_md_table(rows, title=f"总体 Top-{int(args.topk)}（按 accuracy）", max_rows=int(args.topk)))

    if buckets.get("torch_embedding"):
        parts.append(_md_table(buckets["torch_embedding"], title=f"Embedding 系列 Top-{int(args.topk)}", max_rows=int(args.topk)))
    if buckets.get("torch_transfer"):
        parts.append(_md_table(buckets["torch_transfer"], title=f"迁移学习（Transfer）Top-{int(args.topk)}", max_rows=int(args.topk)))
    if buckets.get("classic_numpy"):
        parts.append(_md_table(buckets["classic_numpy"], title=f"传统/手工特征（Numpy）Top-{int(args.topk)}", max_rows=int(args.topk)))

    out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
