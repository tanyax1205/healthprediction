"""Download and load all datasets used by MultiDx.

Every raw file is stored under ``data/raw`` with a SHA-256 checksum recorded in
``data/raw/sources.json``.  If a download fails, the exact manual steps are
printed by :func:`download_all`.

Datasets (all verified reachable without credentials):
  Tier 1 - Kaggle "Disease Symptom Prediction" (itachi9604) - anonymous API
  Tier 2 - UCI Heart Disease (Cleveland), Pima Indians Diabetes,
           UCI ILPD (Kaggle mirror), UCI CKD (rar -> Kaggle zip fallback),
           WDBC (scikit-learn built-in), Kaggle Stroke, UCI hypothyroid
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.config import DATA_RAW, DATA_PROCESSED, ROOT, SEED

# --------------------------------------------------------------------------- #
# Source registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Source:
    url: str
    filename: str  # stored name in data/raw
    kind: str = "file"  # "file" | "zip" | "rar"
    zip_member: str | None = None  # member to extract from zip
    fallback_url: str | None = None
    fallback_filename: str | None = None
    manual: str = ""  # manual download instructions


SOURCES: dict[str, Source] = {
    "symptoms": Source(
        url="https://www.kaggle.com/api/v1/datasets/download/"
        "itachi9604/disease-symptom-description-dataset",
        filename="symptoms.zip",
        kind="zip",
        manual="Kaggle -> https://www.kaggle.com/datasets/itachi9604/"
        "disease-symptom-description-dataset -> Download -> save zip as "
        "data/raw/symptoms.zip",
    ),
    "heart": Source(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data",
        filename="heart_cleveland.data",
        manual="Save https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data as data/raw/heart_cleveland.data",
    ),
    "diabetes": Source(
        url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv",
        filename="pima.csv",
        manual="Save the pima-indians-diabetes.data.csv as data/raw/pima.csv "
        "(mirror of the UCI/NIDDK Pima Indians Diabetes Database).",
    ),
    "liver": Source(
        url="https://www.kaggle.com/api/v1/datasets/download/"
        "uciml/indian-liver-patient-records",
        filename="ilpd.zip",
        kind="zip",
        zip_member="indian_liver_patient.csv",
        fallback_url="https://www.kaggle.com/api/v1/datasets/download/"
        "zubairdhuddi/indian-liver-patient-dataset",
        fallback_filename="ilpd.zip",
        manual="Kaggle -> https://www.kaggle.com/datasets/uciml/"
        "indian-liver-patient-records -> Download -> save as data/raw/ilpd.zip",
    ),
    "ckd": Source(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00336/"
        "Chronic_Kidney_Disease.rar",
        filename="ckd.rar",
        kind="rar",
        fallback_url="https://www.kaggle.com/api/v1/datasets/download/"
        "borschman/early-stage-of-indians-chronic-kidney-diseaseckd",
        fallback_filename="ckd_kaggle.zip",
        manual="Kaggle -> https://www.kaggle.com/datasets/borschman/"
        "early-stage-of-indians-chronic-kidney-diseaseckd -> Download -> "
        "save zip as data/raw/ckd_kaggle.zip",
    ),
    "stroke": Source(
        url="https://www.kaggle.com/api/v1/datasets/download/"
        "fedesoriano/stroke-prediction-dataset",
        filename="stroke.zip",
        kind="zip",
        zip_member="healthcare-dataset-stroke-data.csv",
        manual="Kaggle -> https://www.kaggle.com/datasets/fedesoriano/"
        "stroke-prediction-dataset -> Download -> save as data/raw/stroke.zip",
    ),
    "thyroid": Source(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "thyroid-disease/hypothyroid.data",
        filename="hypothyroid.data",
        manual="Save https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "thyroid-disease/hypothyroid.data as data/raw/hypothyroid.data",
    ),
    # "cancer" (WDBC) is bundled with scikit-learn - no download needed.
}

RAW_PATHS = {k: DATA_RAW / v.filename for k, v in SOURCES.items()}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _http_get(url: str, timeout: int = 60) -> bytes:
    import requests

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "MultiDx/1.0"})
    resp.raise_for_status()
    return resp.content


def _extract_zip(zip_path: Path, member: str | None, dest_dir: Path) -> Path | None:
    """Extract one member (or every text-ish member when *member* is None)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if member is None:
            targets = [n for n in names if Path(n).suffix.lower() in {".csv", ".arff", ".data", ".txt"}]
            targets = targets or names
        else:
            targets = [member] if member in names else [names[0]]
        out: Path | None = None
        for name in targets:
            out = dest_dir / Path(name).name
            out.write_bytes(zf.read(name))
        return out


def _extract_rar(rar_path: Path, dest_dir: Path) -> Path | None:
    """Extract the first .arff from a RAR archive (bsdtar works on macOS/Linux/Win10+)."""
    bsdtar = shutil.which("bsdtar") or shutil.which("tar")
    if bsdtar:
        try:
            subprocess.run(
                [bsdtar, "-xf", str(rar_path), "-C", str(dest_dir)],
                check=True,
                capture_output=True,
                timeout=120,
            )
            arffs = sorted(dest_dir.rglob("*.arff"))
            if arffs:
                # prefer the "full" variant when present
                full = [p for p in arffs if "full" in p.name]
                return full[0] if full else arffs[0]
        except Exception:
            pass
    return None


def download_all(force: bool = False, quiet: bool = False) -> dict[str, dict]:
    """Download every raw source.  Returns a manifest {dataset: {url, sha256, ...}}.

    On failure, prints exact manual download steps and raises FileNotFoundError
    for that dataset (callers decide whether to continue).
    """
    manifest: dict[str, dict] = {}
    manifest_path = DATA_RAW / "sources.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())

    for key, src in SOURCES.items():
        path = RAW_PATHS[key]
        if path.exists() and not force:
            if not quiet:
                print(f"[data] {key}: already present ({path.name})")
            manifest.setdefault(key, {})
            manifest[key].update(
                {"url": src.url, "sha256": _sha256(path), "bytes": path.stat().st_size}
            )
            continue

        urls = [src.url] + ([src.fallback_url] if src.fallback_url else [])
        last_err: Exception | None = None
        for url in urls:
            try:
                if not quiet:
                    print(f"[data] {key}: downloading {url}")
                blob = _http_get(url)
                path.write_bytes(blob)
                if src.kind == "zip":
                    _extract_zip(path, src.zip_member, DATA_RAW)
                elif src.kind == "rar":
                    got = _extract_rar(path, DATA_RAW)
                    if got is None and src.fallback_url is None:
                        raise RuntimeError("RAR extraction failed")
                manifest[key] = {
                    "url": url,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                }
                last_err = None
                break
            except Exception as exc:  # try fallback url
                last_err = exc
                continue

        if last_err is not None:
            # RAR with no extractor: fall back to the Kaggle zip mirror
            if src.fallback_url and src.fallback_filename:
                try:
                    blob = _http_get(src.fallback_url)
                    fb_path = DATA_RAW / src.fallback_filename
                    fb_path.write_bytes(blob)
                    if fb_path.suffix == ".zip":
                        _extract_zip(fb_path, None, DATA_RAW)
                    manifest[key] = {
                        "url": src.fallback_url,
                        "sha256": _sha256(fb_path),
                        "bytes": fb_path.stat().st_size,
                        "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    }
                    last_err = None
                except Exception as exc2:  # pragma: no cover
                    last_err = exc2

        if last_err is not None:
            print(f"[data] ERROR downloading '{key}': {last_err}")
            print(f"[data] MANUAL STEPS for '{key}': {src.manual}")
            raise FileNotFoundError(f"dataset '{key}' could not be downloaded")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

STANDARD_TARGET = "target"


def norm_symptom(name: str) -> str:
    """Normalise a symptom token: 'itching', ' skin_rash', 'dischromic _patches'
    -> 'itching', 'skin rash', 'dischromic patches'."""
    s = str(name).replace("_", " ").strip().lower()
    return " ".join(s.split())


def _finish(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Attach the standard target column and de-duplicate / clean invalid rows."""
    df = df.copy()
    df[STANDARD_TARGET] = target
    df = df.dropna(how="all").drop_duplicates()
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-dataset loaders  ->  each returns (DataFrame with `target` column, meta)
# --------------------------------------------------------------------------- #


def load_symptoms() -> tuple[pd.DataFrame, dict]:
    raw = DATA_RAW / "dataset.csv"
    if not raw.exists():  # zip member name from the Kaggle archive
        raise FileNotFoundError(raw)
    long = pd.read_csv(raw)
    long.columns = [c.strip() for c in long.columns]

    sym_cols = [c for c in long.columns if c.lower().startswith("symptom")]
    long["Disease"] = long["Disease"].astype(str).str.strip()

    records: list[dict] = []
    for _, row in long.iterrows():
        rec: dict = {"target": str(row["Disease"]).strip()}
        for c in sym_cols:
            val = row[c]
            if pd.isna(val):
                continue
            key = norm_symptom(val)
            if key:
                rec[key] = 1
        records.append(rec)
    df = pd.DataFrame(records).fillna(0).astype({c: int for c in pd.DataFrame(records).columns if c != "target"})
    # remove duplicated symptom columns created by inconsistent raw spellings
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop_duplicates().reset_index(drop=True)
    df[STANDARD_TARGET] = df[STANDARD_TARGET].str.strip()

    meta: dict = {"n_raw_rows": int(len(long))}
    sev_path = DATA_RAW / "Symptom-severity.csv"
    if sev_path.exists():
        sev = pd.read_csv(sev_path)
        sev["Symptom"] = sev["Symptom"].map(norm_symptom)
        meta["severity"] = dict(zip(sev["Symptom"], sev["weight"]))
    desc_path = DATA_RAW / "symptom_Description.csv"
    if desc_path.exists():
        desc = pd.read_csv(desc_path)
        desc["Disease"] = desc["Disease"].str.strip()
        meta["description"] = dict(zip(desc["Disease"], desc["Description"]))
    prec_path = DATA_RAW / "symptom_precaution.csv"
    if prec_path.exists():
        prec = pd.read_csv(prec_path)
        prec.columns = [c.strip() for c in prec.columns]
        meta["precautions"] = {
            str(r.iloc[0]).strip(): [str(v) for v in r.iloc[1:] if pd.notna(v)]
            for _, r in prec.iterrows()
        }
    return df, meta


def load_heart() -> tuple[pd.DataFrame, dict]:
    cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
    ]
    df = pd.read_csv(RAW_PATHS["heart"], header=None, names=cols, na_values="?")
    target = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    return _finish(df, target), {"columns": cols[:-1], "n_raw_rows": int(len(df))}


def load_diabetes() -> tuple[pd.DataFrame, dict]:
    cols = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "age", "target",
    ]
    df = pd.read_csv(RAW_PATHS["diabetes"], header=None, names=cols)
    zero_is_missing = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
    n_zero = int((df[zero_is_missing] == 0).sum().sum())
    df[zero_is_missing] = df[zero_is_missing].replace(0, np.nan)
    target = df.pop("target").astype(int)
    return _finish(df, target), {
        "n_raw_rows": int(len(df)),
        "zero_as_missing": zero_is_missing,
        "n_zero_cells_converted": n_zero,
    }


def load_liver() -> tuple[pd.DataFrame, dict]:
    path = DATA_RAW / "indian_liver_patient.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename = {
        "age": "age", "gender": "sex", "total_bilirubin": "total_bilirubin",
        "direct_bilirubin": "direct_bilirubin",
        "alkaline_phosphotase": "alkaline_phosphatase",
        "alamine_aminotransferase": "alt",
        "aspartate_aminotransferase": "ast",
        "total_protiens": "total_proteins",
        "albumin": "albumin", "albumin_and_globulin_ratio": "albumin_globulin_ratio",
    }
    df = df.rename(columns=rename)
    df["sex"] = df["sex"].map({"Male": "M", "Female": "F"})
    target = (df.pop("dataset") == 1).astype(int)  # 1 = liver patient
    return _finish(df, target), {"n_raw_rows": int(len(df))}


def read_arff(path: Path) -> pd.DataFrame:
    """Minimal ARFF reader (attribute names + dense data rows, '?' -> NaN).

    The public UCI CKD file contains 3 malformed rows with an extra field;
    trailing empty fields are dropped and rows that are still inconsistent
    are skipped as invalid records (counted in the returned frame's attrs).
    """
    attrs: list[str] = []
    rows: list[list[str]] = []
    skipped = 0
    in_data = False
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        low = s.lower()
        if low.startswith("@attribute"):
            parts = s.split()
            attrs.append(parts[1].strip("'\""))
        elif low.startswith("@data"):
            in_data = True
        elif in_data:
            fields = [v.strip() for v in s.split(",")]
            while len(fields) > len(attrs) and fields and fields[-1] == "":
                fields.pop()  # trailing comma artefacts
            if len(fields) != len(attrs):
                skipped += 1
                continue
            rows.append(fields)
    df = pd.DataFrame(rows, columns=attrs)
    df.attrs["skipped_invalid_rows"] = skipped
    return df.replace("?", np.nan)


def load_ckd() -> tuple[pd.DataFrame, dict]:
    candidates = [
        DATA_RAW / "chronic_kidney_disease_full.arff",
        DATA_RAW / "chronic_kidney_disease.arff",
    ]
    nested = sorted(DATA_RAW.glob("**/*.arff"))
    path = next((p for p in candidates + nested if p.exists()), None)
    if path is None:
        raise FileNotFoundError("CKD .arff not found under data/raw (see download_all)")
    df = read_arff(path)
    n_skipped = int(df.attrs.get("skipped_invalid_rows", 0))
    keep = [
        "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
        "hemo", "pcv", "htn", "dm", "appet", "ane", "class",
    ]
    missing_cols = [c for c in keep if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CKD arff missing expected columns: {missing_cols}")
    df = df[keep].copy()
    numeric = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # yes/no flags keep NaN when the source value was missing
    df["htn"] = df["htn"].map({"yes": 1.0, "no": 0.0})
    df["dm"] = df["dm"].map({"yes": 1.0, "no": 0.0})
    df["appet"] = df["appet"].map({"good": 1.0, "poor": 0.0})
    df["ane"] = df["ane"].map({"yes": 1.0, "no": 0.0})
    target = (df.pop("class").astype(str).str.lower().str.startswith("ckd")).astype(int)
    return _finish(df, target), {
        "n_raw_rows": int(len(df)),
        "source_arff": str(path.name),
        "n_invalid_rows_skipped": n_skipped,
    }


def load_cancer() -> tuple[pd.DataFrame, dict]:
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer()
    df = pd.DataFrame(bunch.data, columns=[c.replace(" ", "_").lower() for c in bunch.feature_names])
    target = pd.Series((bunch.target == 0).astype(int), name="target")  # 1 = malignant
    return _finish(df, target), {
        "n_raw_rows": int(len(df)),
        "target_names": {"1": "malignant", "0": "benign"},
    }


def load_stroke() -> tuple[pd.DataFrame, dict]:
    path = DATA_RAW / "healthcare-dataset-stroke-data.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df.drop(columns=["id"])
    df = df.rename(columns={"Residence_type": "residence_type"})
    df.columns = [c.lower() for c in df.columns]
    n_before = len(df)
    df = df[df["gender"] != "Other"].copy()  # single invalid/unknown record
    n_dropped = n_before - len(df)
    df["hypertension"] = df["hypertension"].astype(int)
    df["heart_disease"] = df["heart_disease"].astype(int)
    target = df.pop("stroke").astype(int)
    meta = {"n_raw_rows": int(n_before), "n_invalid_dropped": int(n_dropped)}
    return _finish(df, target), meta


THYROID_COLS = [
    "target", "age", "sex",
    "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
    "thyroid_surgery", "query_hypothyroid", "query_hyperthyroid",
    "pregnant", "sick", "tumor", "lithium", "goitre",
    "tsh_measured", "tsh", "t3_measured", "t3",
    "tt4_measured", "tt4", "t4u_measured", "t4u",
    "fti_measured", "fti", "tbg_measured", "tbg",
]


def load_thyroid() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RAW_PATHS["thyroid"], header=None, names=THYROID_COLS, na_values="?")
    n_raw = len(df)
    bool_cols = [
        "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
        "thyroid_surgery", "query_hypothyroid", "query_hyperthyroid",
        "pregnant", "sick", "tumor", "lithium", "goitre",
    ]
    for c in bool_cols:
        df[c] = df[c].map({"t": 1, "f": 0})
    df["sex"] = df["sex"].map({"M": "M", "F": "F"})
    # drop measurement-indicator flags and TBG (mostly unmeasured)
    drop_cols = [c for c in df.columns if c.endswith("_measured")] + ["tbg"]
    df = df.drop(columns=drop_cols)
    num_cols = ["age", "tsh", "t3", "tt4", "t4u", "fti"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    target = (df.pop("target").astype(str).str.lower() == "hypothyroid").astype(int)
    kept = [c for c in df.columns if c != "target"]
    return _finish(df, target), {"n_raw_rows": int(n_raw), "dropped_columns": drop_cols, "kept": kept}


LOADERS: dict[str, Callable[[], tuple[pd.DataFrame, dict]]] = {
    "symptoms": load_symptoms,
    "heart": load_heart,
    "diabetes": load_diabetes,
    "liver": load_liver,
    "ckd": load_ckd,
    "cancer": load_cancer,
    "stroke": load_stroke,
    "thyroid": load_thyroid,
}


def load_task(task: str, use_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """Return (clean dataframe with `target` column, loader metadata)."""
    cache = DATA_PROCESSED / f"{task}.csv"
    meta_cache = DATA_PROCESSED / f"{task}_meta.json"
    if use_cache and cache.exists() and meta_cache.exists():
        df = pd.read_csv(cache)
        meta = json.loads(meta_cache.read_text())
        return df, meta

    if task not in LOADERS:
        raise KeyError(f"unknown task '{task}'")
    if task != "cancer":  # cancer ships inside scikit-learn
        download_all(quiet=True)
    df, meta = LOADERS[task]()
    df.to_csv(cache, index=False)
    meta_cache.write_text(json.dumps(meta, indent=2, default=str))
    return df, meta


def save_processed(df: pd.DataFrame, name: str) -> Path:
    path = DATA_PROCESSED / name
    df.to_csv(path, index=False)
    return path


def manifest_summary() -> pd.DataFrame:
    path = DATA_RAW / "sources.json"
    if not path.exists():
        return pd.DataFrame(columns=["dataset", "url", "bytes", "sha256"])
    man = json.loads(path.read_text())
    rows = [
        {"dataset": k, "url": v.get("url", ""), "bytes": v.get("bytes", 0),
         "sha256": v.get("sha256", "")}
        for k, v in man.items()
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover
    download_all()
    for t in LOADERS:
        df, meta = load_task(t, use_cache=False)
        y = df["target"]
        rate = y.mean() if y.dtype.kind in "iuf" else f"{y.nunique()} classes"
        print(f"{t:10s} rows={len(df):5d} feats={df.shape[1] - 1:3d} target={rate}")
