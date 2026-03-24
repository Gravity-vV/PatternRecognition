from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.task4_runner import ComboSpec, run_task4


@dataclass(frozen=True)
class ResultRow:
    dataset: str
    split_seed: int
    tag: str
    preprocess: str
    feature: str
    classifier: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_time_sec: float
    inference_time_sec: float


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _split_chain(s: str) -> list[str]:
    s = str(s or "").strip()
    if not s or s == "none":
        return []
    return [p for p in s.split("+") if p]


def _parse_transfer_classifier_params(label: str) -> dict[str, Any]:
    """Parse TorchTransferClassifier({...}) label into dict.

    Example label in CSV:
      TorchTransferClassifier({"backbone": "convnext_tiny", ...})
    """

    label = str(label).strip()
    if not label.startswith("TorchTransferClassifier("):
        return {}
    if not label.endswith(")"):
        return {}
    inner = label[len("TorchTransferClassifier(") : -1].strip()
    try:
        obj = json.loads(inner)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _pick_best_embedding_logreg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cand: list[dict[str, Any]] = []
    for r in rows:
        if str(r.get("dataset", "")) != "oxford_pets":
            continue
        if str(r.get("classifier", "")) != "logreg":
            continue
        feat = str(r.get("feature", ""))
        if "convnext_tiny_embedding" not in feat:
            continue
        tag = str(r.get("tag", ""))
        if "sweep_torch_embed" not in tag:
            continue
        cand.append(r)

    if not cand:
        raise RuntimeError("no embedding+logreg rows found in leaderboard")

    cand.sort(key=lambda x: _to_float(x.get("accuracy", 0.0)), reverse=True)
    return cand[0]


def _pick_best_transfer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cand: list[dict[str, Any]] = []
    for r in rows:
        if str(r.get("dataset", "")) != "oxford_pets":
            continue
        tag = str(r.get("tag", ""))
        if not tag.startswith("transfer_"):
            continue
        cand.append(r)

    if not cand:
        # fallback: just take best by accuracy
        cand = [r for r in rows if str(r.get("dataset", "")) == "oxford_pets"]

    if not cand:
        raise RuntimeError("no transfer rows found")

    cand.sort(key=lambda x: _to_float(x.get("accuracy", 0.0)), reverse=True)
    return cand[0]


def _combo_from_embedding_row(best: dict[str, Any], *, split_seed: int, pretrained: bool) -> tuple[ComboSpec, str]:
    preprocess = _split_chain(str(best.get("preprocess", "")))
    features = _split_chain(str(best.get("feature", "")))
    if not features or features[0] != "convnext_tiny_embedding":
        raise RuntimeError(f"unexpected embedding feature chain: {features}")

    classifier = "logreg"

    params: dict[str, dict[str, Any]] = {
        "convnext_tiny_embedding": {"seed": int(split_seed), "pretrained": bool(pretrained)},
        "logreg": {"random_state": int(split_seed)},
    }
    if "pca" in features:
        params["pca"] = {"n_components": 128, "random_state": int(split_seed)}

    combo = ComboSpec(
        tag="ms_best_embedding_logreg",
        preprocess=preprocess,
        features=features,
        classifier=classifier,
        params=params,
    )
    return combo, str(best.get("tag", ""))


def _combo_from_transfer_row(best: dict[str, Any]) -> tuple[ComboSpec, dict[str, Any]]:
    preprocess = _split_chain(str(best.get("preprocess", "")))
    params = _parse_transfer_classifier_params(str(best.get("classifier", "")))

    # Only keep the minimal knobs we want to reproduce; seed/device/batch_size will be filled by runner.
    keep: dict[str, Any] = {}
    for k in ("mode", "epochs", "lr"):
        if k in params:
            keep[k] = params[k]

    combo = ComboSpec(
        tag="ms_best_transfer",
        preprocess=preprocess,
        features=[],
        classifier="convnext_tiny_transfer",
        params={"convnext_tiny_transfer": keep},
    )
    return combo, keep


def _rows_from_csv(path: Path) -> list[ResultRow]:
    out: list[ResultRow] = []
    for r in _read_csv(path):
        out.append(
            ResultRow(
                dataset=str(r.get("dataset", "")),
                split_seed=int(r.get("split_seed", 0)),
                tag=str(r.get("tag", "")),
                preprocess=str(r.get("preprocess", "")),
                feature=str(r.get("feature", "")),
                classifier=str(r.get("classifier", "")),
                accuracy=_to_float(r.get("accuracy", 0.0)),
                precision=_to_float(r.get("precision", 0.0)),
                recall=_to_float(r.get("recall", 0.0)),
                f1=_to_float(r.get("f1", 0.0)),
                train_time_sec=_to_float(r.get("train_time_sec", 0.0)),
                inference_time_sec=_to_float(r.get("inference_time_sec", 0.0)),
            )
        )
    return out


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1))


def _fmt_mean_std(mean: float, std: float, *, digits: int = 4) -> str:
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def _write_multiseed_md(
    out_md: Path,
    *,
    rows: list[ResultRow],
    embedding_source_tag: str,
    transfer_source_tag: str,
    transfer_params: dict[str, Any],
    seeds: list[int],
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)

    def _group_key(r: ResultRow) -> str:
        if r.tag == "ms_best_embedding_logreg":
            return "Embedding+LogReg（best）"
        if r.tag == "ms_best_transfer":
            return "Transfer（best, 5 epochs）"
        return r.tag

    groups: dict[str, list[ResultRow]] = {}
    for r in rows:
        groups.setdefault(_group_key(r), []).append(r)

    lines: list[str] = []
    lines.append("# Multi-seed 稳定性补充实验（best config × 3 seeds）")
    lines.append("")
    lines.append(f"- seeds: {seeds}")
    lines.append(f"- embedding 来源 tag（seed=0 的 sweep 最优 logreg）: {embedding_source_tag}")
    lines.append(f"- transfer 来源 tag（seed=0 的 5 epochs 最优）: {transfer_source_tag}")
    if transfer_params:
        lines.append(f"- transfer 复现参数（关键项）: mode={transfer_params.get('mode')}, epochs={transfer_params.get('epochs')}, lr={transfer_params.get('lr')}")
    lines.append("")

    # Per-seed table
    lines.append("## 分 seed 结果")
    lines.append("")
    lines.append("|方法|split_seed|acc|precision|recall|f1|train(s)|infer(s)|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("Embedding+LogReg（best）", "Transfer（best, 5 epochs）"):
        rs = sorted(groups.get(name, []), key=lambda x: x.split_seed)
        for r in rs:
            lines.append(
                f"|{name}|{r.split_seed}|{r.accuracy:.4f}|{r.precision:.4f}|{r.recall:.4f}|{r.f1:.4f}|{r.train_time_sec:.2f}|{r.inference_time_sec:.3f}|"
            )

    # Summary
    lines.append("")
    lines.append("## 汇总（均值±标准差）")
    lines.append("")
    lines.append("|方法|acc|precision|recall|f1|train(s)|infer(s)|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for name in ("Embedding+LogReg（best）", "Transfer（best, 5 epochs）"):
        rs = groups.get(name, [])
        acc_m, acc_s = _mean_std([x.accuracy for x in rs])
        pre_m, pre_s = _mean_std([x.precision for x in rs])
        rec_m, rec_s = _mean_std([x.recall for x in rs])
        f1_m, f1_s = _mean_std([x.f1 for x in rs])
        tr_m, tr_s = _mean_std([x.train_time_sec for x in rs])
        inf_m, inf_s = _mean_std([x.inference_time_sec for x in rs])
        lines.append(
            "|"
            + name
            + "|"
            + _fmt_mean_std(acc_m, acc_s, digits=4)
            + "|"
            + _fmt_mean_std(pre_m, pre_s, digits=4)
            + "|"
            + _fmt_mean_std(rec_m, rec_s, digits=4)
            + "|"
            + _fmt_mean_std(f1_m, f1_s, digits=4)
            + "|"
            + _fmt_mean_std(tr_m, tr_s, digits=2)
            + "|"
            + _fmt_mean_std(inf_m, inf_s, digits=3)
            + "|"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run multi-seed stability check for best embedding+LogReg and best transfer configs")
    p.add_argument("--seeds", default="0,1,2", help="comma-separated split_seeds, e.g. 0,1,2")
    p.add_argument(
        "--embedding-leaderboard",
        default="experiments/sweep_all_leaderboard.csv",
        help="leaderboard CSV used to pick best embedding+logreg config",
    )
    p.add_argument(
        "--transfer-raw",
        default="experiments/transfer_more_epochs_raw.csv",
        help="raw CSV used to pick best transfer (typically 5 epochs) config",
    )
    p.add_argument("--out-csv", default="experiments/multiseed_best_raw.csv")
    p.add_argument("--out-dir", default="experiments/artifacts_task4_multiseed")
    p.add_argument("--out-md", default="report/assets/multiseed_summary.md")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)

    args = p.parse_args(argv)

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if not seeds:
        raise SystemExit("--seeds is empty")

    emb_rows = _read_csv(Path(str(args.embedding_leaderboard)))
    tr_rows = _read_csv(Path(str(args.transfer_raw)))

    best_emb = _pick_best_embedding_logreg(emb_rows)
    best_tr = _pick_best_transfer(tr_rows)

    transfer_combo, transfer_params = _combo_from_transfer_row(best_tr)

    all_rows: list[dict[str, Any]] = []
    tmp_dir = Path("experiments/_tmp_multiseed")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        emb_combo, emb_source_tag = _combo_from_embedding_row(best_emb, split_seed=seed, pretrained=bool(args.pretrained))

        # Important: make per-seed outputs, then merge.
        out_csv_seed = tmp_dir / f"seed_{seed}.csv"
        run_task4(
            split_seed=int(seed),
            epochs=5,  # ensure transfer uses 5 even if params were missing
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            pretrained=bool(args.pretrained),
            out_csv=out_csv_seed,
            out_dir=Path(str(args.out_dir)),
            combos=[emb_combo, transfer_combo],
        )
        all_rows.extend(_read_csv(out_csv_seed))

    # Write merged CSV
    out_csv = Path(str(args.out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not all_rows:
        raise SystemExit("no rows produced")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # Write markdown summary
    out_md = Path(str(args.out_md))
    parsed = _rows_from_csv(out_csv)
    _write_multiseed_md(
        out_md,
        rows=parsed,
        embedding_source_tag=str(best_emb.get("tag", "")),
        transfer_source_tag=str(best_tr.get("tag", "")),
        transfer_params=transfer_params,
        seeds=seeds,
    )

    print(f"wrote: {out_csv}")
    print(f"wrote: {out_md}")


if __name__ == "__main__":
    main()
