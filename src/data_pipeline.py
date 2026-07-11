"""
data_pipeline.py
=================
Loads the base tabular supply-chain dataset and builds a MULTIMODAL dataset
around it, as required by the hackathon brief:

  1. TABULAR      -> the raw Kaggle-style supply_chain_data.csv (given)
  2. TIME SERIES  -> a synthetic 12-week lead-time / stock-level history per
                      SKU, generated from the row's own statistics (so the
                      series is internally consistent with the tabular row)
  3. TEXT         -> an auto-generated supplier / QA note per SKU, built from
                      a template bank that is conditioned on the row's real
                      inspection result / defect rate / shipping carrier, so
                      the language genuinely correlates with risk
  4. IMAGE        -> a small "risk gauge" PNG per SKU rendered with
                      matplotlib, encoding defect-rate + inspection result
                      as a visual gauge/heatmap (stand-in for a QC photo of
                      the shipment). A CNN is trained to read the gauge back.

Why synthetic text/image/time-series instead of scraping real ones?
The Kaggle source only ships the tabular table. Hackathon rules allow
multimodal *data*, not necessarily multimodal *sources* — generating
label-consistent synthetic modalities from the ground-truth tabular row is
a standard, transparent way to satisfy the requirement without fabricating
false claims about the world. This is documented in README.md and should be
disclosed in the demo/presentation.

Run directly to (re)build everything:
    python src/data_pipeline.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG_SEED = 42
np.random.seed(RNG_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(BASE_DIR, "data", "supply_chain_data.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "synthetic_assets", "images")
TIMESERIES_JSON = os.path.join(BASE_DIR, "synthetic_assets", "timeseries.json")
TEXT_CSV = os.path.join(BASE_DIR, "synthetic_assets", "supplier_notes.csv")
ENRICHED_CSV = os.path.join(BASE_DIR, "data", "enriched_dataset.csv")

RISK_CLASSES = ["Low", "Medium", "High"]

# ---------------------------------------------------------------------------
# 1. LOAD + LABEL
# ---------------------------------------------------------------------------

def load_base(path=DATA_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def engineer_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """Builds a ground-truth Risk Score (0-100) and 3-class Risk Label from
    real columns already in the dataset. This is the target the deep
    learning models learn to predict.
    """
    df = df.copy()

    def norm(s):
        s = s.astype(float)
        rng = (s.max() - s.min())
        return (s - s.min()) / rng if rng > 0 else s * 0

    inspection_penalty = df["Inspection results"].map(
        {"Fail": 1.0, "Pending": 0.5, "Pass": 0.0}
    ).fillna(0.5)

    score = (
        0.30 * norm(df["Defect rates"]) +
        0.20 * inspection_penalty +
        0.20 * norm(df["Lead times"]) +
        0.15 * norm(df["Shipping times"]) +
        0.15 * (1 - norm(df["Stock levels"]))
    ) * 100

    df["Risk Score"] = score.round(2)
    df["Risk Label"] = pd.cut(
        df["Risk Score"], bins=[-1, 33, 66, 101], labels=RISK_CLASSES
    ).astype(str)
    return df


# ---------------------------------------------------------------------------
# 2. TIME SERIES MODALITY
# ---------------------------------------------------------------------------

def generate_time_series(df: pd.DataFrame, weeks: int = 12) -> dict:
    """Synthesizes a 12-week lead-time trend per SKU: a random walk that is
    anchored to the row's real 'Lead times' value and drifts upward for
    high-risk SKUs (simulating worsening supplier performance) and stays
    flat/improves for low-risk SKUs. This is the sequence fed to the LSTM.
    """
    series_by_sku = {}
    for _, row in df.iterrows():
        base = float(row["Lead times"])
        risk = row["Risk Score"] / 100.0
        drift = (risk - 0.4) * 0.6          # trending worse if risky
        noise = np.random.normal(0, 1.0, size=weeks)
        walk = np.cumsum(np.random.normal(drift, 0.6, size=weeks))
        series = np.clip(base + walk + noise, 1, None).round(2)
        series_by_sku[row["SKU"]] = series.tolist()

    with open(TIMESERIES_JSON, "w") as f:
        json.dump(series_by_sku, f)
    return series_by_sku


# ---------------------------------------------------------------------------
# 3. TEXT MODALITY
# ---------------------------------------------------------------------------

_HIGH_RISK_PHRASES = [
    "Shipment flagged after QA inspection {inspection}; defect rate elevated at {defect:.1f}%.",
    "Supplier {supplier} reported delays; lead time stretched to {lead} days on route {route}.",
    "Carrier {carrier} noted handling issues; customer complaints trending up for SKU {sku}.",
    "Recurrent quality escalations from {location} facility, inspection result: {inspection}.",
]
_MED_RISK_PHRASES = [
    "Shipment on schedule but defect rate ({defect:.1f}%) worth monitoring next cycle.",
    "Supplier {supplier} within tolerance; minor variance in lead time ({lead} days).",
    "Inspection result {inspection} for SKU {sku}; no corrective action required yet.",
    "Carrier {carrier} performing near SLA on route {route}; watch stock levels.",
]
_LOW_RISK_PHRASES = [
    "Clean inspection ({inspection}) for SKU {sku}; defect rate low at {defect:.1f}%.",
    "Supplier {supplier} consistently on-time via {carrier}; no open issues.",
    "Stock levels healthy, lead time steady at {lead} days on route {route}.",
    "No quality escalations reported from {location} facility this cycle.",
]


def generate_supplier_notes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        bank = (
            _HIGH_RISK_PHRASES if row["Risk Label"] == "High" else
            _MED_RISK_PHRASES if row["Risk Label"] == "Medium" else
            _LOW_RISK_PHRASES
        )
        template = np.random.choice(bank)
        note = template.format(
            inspection=row["Inspection results"],
            defect=row["Defect rates"],
            supplier=row["Supplier name"],
            lead=int(row["Lead times"]),
            route=row["Routes"],
            carrier=row["Shipping carriers"],
            sku=row["SKU"],
            location=row["Location"],
        )
        rows.append({"SKU": row["SKU"], "supplier_note": note})
    text_df = pd.DataFrame(rows)
    text_df.to_csv(TEXT_CSV, index=False)
    return text_df


# ---------------------------------------------------------------------------
# 4. IMAGE MODALITY  (risk-gauge PNG per SKU -> CNN input)
# ---------------------------------------------------------------------------

def _risk_color(score):
    if score < 33:
        return "#2e7d32"    # green
    elif score < 66:
        return "#f9a825"    # amber
    return "#c62828"        # red


def generate_gauge_image(sku, score, inspection, path):
    fig, ax = plt.subplots(figsize=(2, 2), dpi=64)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")

    theta = np.linspace(np.pi, 0, 100)
    ax.plot(np.cos(theta), np.sin(theta), color="#dddddd", linewidth=14, solid_capstyle="round")

    frac = np.clip(score / 100.0, 0, 1)
    theta_val = np.linspace(np.pi, np.pi - frac * np.pi, 60)
    ax.plot(np.cos(theta_val), np.sin(theta_val), color=_risk_color(score),
             linewidth=14, solid_capstyle="round")

    ax.text(0, -0.35, f"{score:.0f}", ha="center", va="center", fontsize=22, fontweight="bold")
    ax.text(0, -0.75, inspection, ha="center", va="center", fontsize=10, color="#555555")

    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def generate_images(df: pd.DataFrame, out_dir=IMAGES_DIR):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for _, row in df.iterrows():
        p = os.path.join(out_dir, f"{row['SKU']}.png")
        generate_gauge_image(row["SKU"], row["Risk Score"], row["Inspection results"], p)
        paths[row["SKU"]] = p
    return paths


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

def build_all(save=True):
    df = load_base()
    df = engineer_risk_label(df)
    ts = generate_time_series(df)
    text_df = generate_supplier_notes(df)
    img_paths = generate_images(df)

    df["image_path"] = df["SKU"].map(img_paths)
    df = df.merge(text_df, on="SKU", how="left")

    if save:
        df.to_csv(ENRICHED_CSV, index=False)

    print(f"Loaded {len(df)} rows.")
    print(f"Risk label distribution:\n{df['Risk Label'].value_counts()}")
    print(f"Wrote enriched tabular data -> {ENRICHED_CSV}")
    print(f"Wrote {len(img_paths)} gauge images -> {IMAGES_DIR}")
    print(f"Wrote time-series JSON -> {TIMESERIES_JSON}")
    print(f"Wrote supplier notes -> {TEXT_CSV}")
    return df, ts


if __name__ == "__main__":
    build_all()
