"""Export evaluation results as CSV and LaTeX for the IEEE paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.config import ARTIFACTS_DIR, MODEL_LABELS, REPORTS_DIR, TASKS

__all__ = ["results_frame", "export"]

_METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]


def _load_latest(task: str) -> dict | None:
    idx = ARTIFACTS_DIR / "index.json"
    if not idx.exists():
        return None
    index = json.loads(idx.read_text())
    version = index.get(task)
    if not version:
        return None
    meta = ARTIFACTS_DIR / task / version / "metadata.json"
    if not meta.exists():
        return None
    return json.loads(meta.read_text())


def results_frame() -> pd.DataFrame:
    rows = []
    for task in TASKS:
        meta = _load_latest(task)
        if not meta:
            continue
        m = meta["metrics"]
        row = {
            "task": task,
            "label": TASKS[task].label,
            "best_model": MODEL_LABELS.get(meta["best_model"], meta["best_model"]),
            "cv_score": round(meta["cv_score"], 4),
            "brier": round(m.get("brier", float("nan")), 4),
        }
        for col in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "f1_macro"):
            if col in m:
                row[col] = round(float(m[col]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _latex(df: pd.DataFrame) -> str:
    cols = ["task", "best_model", "cv_score", "accuracy", "recall", "f1", "roc_auc", "pr_auc", "brier"]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].copy()
    header = " & ".join(c.replace("_", "\\_").title() for c in cols) + " \\\\"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{MultiDx baseline model performance per task (calibrated, held-out test).}",
        "\\label{tab:multidx-results}",
        "\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for _, r in sub.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v).replace("_", "\\_"))
        lines.append(" & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def export() -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = results_frame()
    csv_path = REPORTS_DIR / "results.csv"
    tex_path = REPORTS_DIR / "results.tex"
    df.to_csv(csv_path, index=False)
    tex_path.write_text(_latex(df))
    return csv_path, tex_path


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="Export results table").parse_args(argv)
    csv_path, tex_path = export()
    print(f"wrote {csv_path}")
    print(f"wrote {tex_path}")
    print(results_frame().to_string(index=False))


if __name__ == "__main__":
    main()
