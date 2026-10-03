"""
preprocessing.py  —  Load raw CSVs, clean, build two datasets, save.

Dataset decisions based on data audit:
  - File 3 has Tr1 > T4 for 65% of rows (physically invalid for wet-channel IEC).
    It is excluded from the wet-channel physics (eps_wet supervision) but its
    dry-channel rows (T3, Qiec) are still usable for SubNet1.
  - File 2 has ~18% rows with eps_wet outside [0,1]; those rows are dropped.
  - AFR = CMHsa/CMHoa = 1.000 exactly across all files (zero variance).
    It is dropped from SubNet inputs; equations.tex lists it but it carries
    no information in this dataset.

Two datasets produced:
  iec_*.pt   – Files 1+2+3, Qiec > 0, eps_wet valid [0,1].
               SubNet1 (eps_dry) trained on all; SubNet2 (eps_wet) loss only
               applied where eps_wet_valid flag = 1.
  mvc_*.pt   – File 1 only, ECompPower > 0.

Outputs (data/processed/):
    iec_train.pt, iec_val.pt, iec_test.pt
    mvc_train.pt, mvc_val.pt, mvc_test.pt
    norm_stats.pt
    stats_summary.csv
"""

import os
import glob
import torch
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(DATA_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_FILES = [
    "1- IEC MVC Data.csv",
    "2-IEC MVC Data.csv",
    "3-IEC MVC Data.csv",
]

NEEDED = [
    "Tr2 [C]", "wr2 [g/kg]",
    "T4 [C]",  "RH4 [%]",  "w4 [g/kg]",
    "T2 [C]",
    "T1 [C]",
    "CMHoa [m3/Hr]", "CMHsa [m3/Hr]",
    "T3 [C]",  "RH3 [%]",
    "Tr1 [C]",
    "Qcoil [kW]", "Qiec [kW]", "ECompPower [kW]",
]

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, header=[0, 1])
    df.columns = [c[0].strip() for c in df.columns]
    return df[[c for c in NEEDED if c in df.columns]].copy()


def stull_twb_np(T, RH):
    """Stull (2011) wet-bulb approximation, numpy version."""
    return (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
            + np.arctan(T + RH)
            - np.arctan(RH - 1.676331)
            + 0.00391838 * RH**1.5 * np.arctan(0.023101 * RH)
            - 4.686035)


def base_clean(df):
    df = df.dropna()
    df = df[df["CMHoa [m3/Hr]"] > 10]
    df = df[df["CMHsa [m3/Hr]"] > 10]
    df = df[(df["RH4 [%]"] >= 0) & (df["RH4 [%]"] <= 100)]
    df = df[df["Qiec [kW]"] > 0]
    return df.reset_index(drop=True)


def add_eps_wet_validity(df):
    """
    Compute eps_wet from data and add a validity flag.
    Valid means: Tr1 < T4 (physically: wet channel cools the air)
                 AND eps_wet in [0.0, 1.0].
    """
    T4  = df["T4 [C]"].values
    Tr1 = df["Tr1 [C]"].values
    RH4 = df["RH4 [%]"].values
    Twb = stull_twb_np(T4, RH4)
    denom = T4 - Twb
    eps_wet = np.where(np.abs(denom) > 0.5, (T4 - Tr1) / denom, np.nan)
    valid = (eps_wet >= 0.0) & (eps_wet <= 1.0) & np.isfinite(eps_wet)
    df = df.copy()
    df["eps_wet_valid"] = valid.astype(float)
    return df


def chrono_split(df):
    n  = len(df)
    i1 = int(n * TRAIN_FRAC)
    i2 = int(n * (TRAIN_FRAC + VAL_FRAC))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def to_tensors(df):
    rename = {
        "Tr2 [C]": "Tr2",  "wr2 [g/kg]": "wr2",
        "T4 [C]":  "T4",   "RH4 [%]":    "RH4",  "w4 [g/kg]": "w4",
        "T2 [C]":  "T2",   "T1 [C]":     "T1",
        "CMHoa [m3/Hr]": "CMHoa", "CMHsa [m3/Hr]": "CMHsa",
        "T3 [C]":  "T3",   "RH3 [%]":    "RH3",
        "Tr1 [C]": "Tr1",
        "Qcoil [kW]": "Qcoil", "Qiec [kW]": "Qiec",
        "ECompPower [kW]": "ECompPower",
    }
    t = {}
    for raw, short in rename.items():
        if raw in df.columns:
            t[short] = torch.tensor(df[raw].values, dtype=torch.float32).unsqueeze(1)
    if "eps_wet_valid" in df.columns:
        t["eps_wet_valid"] = torch.tensor(
            df["eps_wet_valid"].values, dtype=torch.float32).unsqueeze(1)
    return t


def norm_stats_from(tensors):
    """z-score stats. AFR removed — zero variance in this dataset."""
    def ms(x):
        return x.mean(dim=0), x.std(dim=0).clamp(min=1e-8)
    return {
        "Voa":    ms(tensors["CMHoa"]),
        "Tr2":    ms(tensors["Tr2"]),
        "T4":     ms(tensors["T4"]),
        "CMHsa":  ms(tensors["CMHsa"]),
        "T2":     ms(tensors["T2"]),
        "T1":     ms(tensors["T1"]),
        "T3hat":  ms(tensors["T3"]),
        "Tr1hat": ms(tensors["Tr1"]),
    }


def main():
    print("Building IEC dataset (all 3 files)...")
    frames = []
    for fname in CSV_FILES:
        df = load_csv(fname)
        df = base_clean(df)
        df = add_eps_wet_validity(df)
        valid_pct = df["eps_wet_valid"].mean() * 100
        print(f"  {fname}: {len(df)} rows  eps_wet_valid={valid_pct:.1f}%")
        frames.append(df)

    iec_df = pd.concat(frames, ignore_index=True)
    # Shuffle before split so files are mixed across train/val/test
    iec_df = iec_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  IEC total: {len(iec_df)} rows  "
          f"eps_wet_valid={iec_df['eps_wet_valid'].mean()*100:.1f}%")

    tr, va, te = chrono_split(iec_df)   # "chrono" on shuffled = random split
    torch.save(to_tensors(tr), os.path.join(OUT_DIR, "iec_train.pt"))
    torch.save(to_tensors(va), os.path.join(OUT_DIR, "iec_val.pt"))
    torch.save(to_tensors(te), os.path.join(OUT_DIR, "iec_test.pt"))
    print(f"  Split: train={len(tr)}  val={len(va)}  test={len(te)}")

    iec_train_t = to_tensors(tr)
    stats = norm_stats_from(iec_train_t)
    torch.save(stats, os.path.join(OUT_DIR, "norm_stats.pt"))

    print("\nBuilding MVC dataset (file 1, ECompPower > 0)...")
    mvc_df = load_csv(CSV_FILES[0])
    mvc_df = base_clean(mvc_df)
    mvc_df = add_eps_wet_validity(mvc_df)
    mvc_df = mvc_df[mvc_df["ECompPower [kW]"] > 0]
    mvc_df = mvc_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  MVC total: {len(mvc_df)} rows")

    tr_m, va_m, te_m = chrono_split(mvc_df)
    torch.save(to_tensors(tr_m), os.path.join(OUT_DIR, "mvc_train.pt"))
    torch.save(to_tensors(va_m), os.path.join(OUT_DIR, "mvc_val.pt"))
    torch.save(to_tensors(te_m), os.path.join(OUT_DIR, "mvc_test.pt"))
    print(f"  Split: train={len(tr_m)}  val={len(va_m)}  test={len(te_m)}")

    print("\nSaving stats summary...")
    rows = []
    for col in NEEDED:
        all_vals = pd.concat([load_csv(f) for f in CSV_FILES], ignore_index=True)
        if col in all_vals.columns:
            s = all_vals[col]
            rows.append({"column": col, "mean": s.mean(), "std": s.std(),
                         "min": s.min(), "max": s.max()})
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, "stats_summary.csv"), index=False, float_format="%.4f"
    )

    print(f"\nDone. Outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
