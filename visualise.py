"""
visualise.py  —  Plots from a completed training run.

Usage:
    python visualise.py runs/Logs-HH-MM-DD-Mon-YYYY/
    python visualise.py runs/Logs-HH-MM-DD-Mon-YYYY/ --show   # also open windows

Plots saved as PNGs inside the run directory:
    loss_curves.png         – total & per-term train/val loss over epochs
    parity_T3.png           – predicted vs measured T3 (test set)
    parity_Tr1.png          – predicted vs measured Tr1
    parity_T1.png           – predicted vs measured T1
    parity_Qcoil.png        – predicted vs measured Qcoil
    parity_Qiec.png         – predicted vs measured Qiec
    parity_ECompPower.png   – predicted vs measured ECompPower
    effectiveness.png       – ε_dry and ε_wet: predicted vs physics-derived
    residuals.png           – residual distributions for all primary targets
    error_boxplot.png       – absolute error boxplots across targets
    cop_distribution.png    – histogram of predicted COP (derived output)
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")        # non-interactive backend by default
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})

BLUE, ORANGE, GREEN, RED = "#3A7BD5", "#E07B39", "#2ECC71", "#E74C3C"


# ── Helpers ───────────────────────────────────────────────────────────────────

def savefig(fig, path, show):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {os.path.basename(path)}")
    if show:
        plt.show()
    plt.close(fig)


def load_history(run_dir):
    with open(os.path.join(run_dir, "history.json")) as f:
        h = json.load(f)
    return h["train"], h["val"]


def load_predictions(run_dir, mode):
    path = os.path.join(run_dir, f"{mode}_test_predictions.pt")
    return torch.load(path, map_location="cpu")


def t(tensor):
    """Flatten to numpy 1-D."""
    return tensor.squeeze().numpy()


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))

def bias(a, b):
    return float(np.mean(b - a))

def r2_score(a, b):
    var_a = np.var(a)
    if var_a > 1e-12:
        return float(1 - np.var(a - b) / var_a)
    return float("nan")

# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_loss_curves(train_h, val_h, run_dir, show):
    terms = ["total", "T3", "Tr1", "RH3", "Qiec", "eps_dry", "eps_wet",
             "T1", "Qcoil", "ECompPower"]
    labels = {"total": "Total", "T3": "T3", "Tr1": "Tr1", "RH3": "RH3",
              "Qiec": "Qiec", "eps_dry": "ε_dry", "eps_wet": "ε_wet",
              "T1": "T1", "Qcoil": "Qcoil", "ECompPower": "ECompPower"}

    epochs = range(1, len(train_h) + 1)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()

    for ax, term in zip(axes, terms):
        tr_vals = [d.get(term, float("nan")) for d in train_h]
        va_vals = [d.get(term, float("nan")) for d in val_h]
        ax.plot(epochs, tr_vals, color=BLUE,   lw=1.5, label="Train")
        ax.plot(epochs, va_vals, color=ORANGE, lw=1.5, label="Val",  ls="--")
        ax.set_title(labels[term])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)
        # Log scale if range is large
        tr_clean = [v for v in tr_vals if not np.isnan(v) and v > 0]
        if tr_clean and (max(tr_clean) / max(min(tr_clean), 1e-10)) > 100:
            ax.set_yscale("log")

    fig.suptitle("Training & Validation Loss Curves", fontsize=14, fontweight="bold")
    savefig(fig, os.path.join(run_dir, "loss_curves.png"), show)


def plot_parity(y_true, y_pred, label, units, run_dir, show, fname=None):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, "k--", lw=1, label="Perfect")
    ax.scatter(y_true, y_pred, s=6, alpha=0.35, color=BLUE, rasterized=True)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f"Measured {label} [{units}]")
    ax.set_ylabel(f"Predicted {label} [{units}]")
    ax.set_title(f"{label} — Parity Plot\n"
                 f"RMSE={rmse(y_true, y_pred):.3f}  MAE={mae(y_true, y_pred):.3f} {units}")
    fname = fname or f"parity_{label.replace(' ','_')}.png"
    savefig(fig, os.path.join(run_dir, fname), show)


def plot_effectiveness(iec_data, run_dir, show):
    pred  = iec_data["predictions"]
    inp   = iec_data["inputs"]

    eps_dry_pred = t(pred["hat_eps_dry"])
    eps_dry_obs  = t(pred["eps_dry_obs"])
    eps_wet_pred = t(pred["hat_eps_wet"])
    eps_wet_obs  = t(pred["eps_wet_obs"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, pred_v, obs_v, name in [
        (axes[0], eps_dry_pred, eps_dry_obs, "ε_dry"),
        (axes[1], eps_wet_pred, eps_wet_obs, "ε_wet"),
    ]:
        lo = min(pred_v.min(), obs_v.min())
        hi = max(pred_v.max(), obs_v.max())
        pad = (hi - lo) * 0.05
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, "k--", lw=1)
        ax.scatter(obs_v, pred_v, s=6, alpha=0.35, color=GREEN, rasterized=True)
        ax.set_xlabel(f"{name} (physics-derived from data)")
        ax.set_ylabel(f"{name} (SubNet prediction)")
        ax.set_title(f"{name}  RMSE={rmse(obs_v, pred_v):.4f}")
        ax.set_xlim(lims); ax.set_ylim(lims)

    fig.suptitle("Effectiveness: SubNet Prediction vs. Physics-Derived Target",
                 fontweight="bold")
    savefig(fig, os.path.join(run_dir, "effectiveness.png"), show)


def plot_residuals(targets_dict, run_dir, show):
    """targets_dict: {label: (y_true, y_pred, units)}"""
    n = len(targets_dict)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 7))
    axes = axes.flatten()

    for ax, (label, (y_true, y_pred, units)) in zip(axes, targets_dict.items()):
        res = y_pred - y_true
        ax.hist(res, bins=50, color=BLUE, alpha=0.75, edgecolor="white", lw=0.3)
        ax.axvline(0, color=RED, lw=1.5, ls="--")
        ax.set_title(f"{label}\nMean={res.mean():.3f}  Std={res.std():.3f}")
        ax.set_xlabel(f"Residual [{units}]")
        ax.set_ylabel("Count")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Prediction Residual Distributions (Test Set)", fontweight="bold")
    savefig(fig, os.path.join(run_dir, "residuals.png"), show)


def plot_error_boxplot(targets_dict, run_dir, show):
    labels, data = [], []
    for label, (y_true, y_pred, units) in targets_dict.items():
        labels.append(f"{label}\n[{units}]")
        data.append(np.abs(y_pred - y_true))

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", lw=2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Absolute Error")
    ax.set_title("Absolute Error Distribution per Target (Test Set)", fontweight="bold")
    savefig(fig, os.path.join(run_dir, "error_boxplot.png"), show)


def plot_cop_distribution(mvc_data, run_dir, show):
    pred = mvc_data["predictions"]
    cop  = t(pred["hat_COP"])
    # Remove extreme outliers for display
    q1, q99 = np.percentile(cop, 1), np.percentile(cop, 99)
    cop_clip = cop[(cop >= q1) & (cop <= q99)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cop_clip, bins=60, color=ORANGE, alpha=0.8, edgecolor="white", lw=0.3)
    ax.axvline(cop_clip.mean(), color=RED, lw=2, ls="--",
               label=f"Mean = {cop_clip.mean():.2f}")
    ax.set_xlabel("Predicted COP_system")
    ax.set_ylabel("Count")
    ax.set_title("System COP Distribution (Test Set, 1st–99th pctile)", fontweight="bold")
    ax.legend()
    savefig(fig, os.path.join(run_dir, "cop_distribution.png"), show)


def plot_timeseries(targets_dict, run_dir, show, n_samples=500):
    """Plot the first n_samples of test set as a time-series comparison."""
    n = len(targets_dict)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)

    for ax, (label, (y_true, y_pred, units)) in zip(axes, targets_dict.items()):
        idx = np.arange(min(len(y_true), n_samples))
        ax.plot(idx, y_true[:len(idx)], color=BLUE,   lw=1.2, label="Measured", alpha=0.8)
        ax.plot(idx, y_pred[:len(idx)], color=ORANGE, lw=1.2, label="Predicted", ls="--", alpha=0.9)
        ax.set_ylabel(f"{label} [{units}]")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel(f"Sample index (first {n_samples} of test set)")
    fig.suptitle("Measured vs. Predicted — Test Set Time Series", fontweight="bold")
    savefig(fig, os.path.join(run_dir, "timeseries.png"), show)


def save_metrics_txt(iec_data, mvc_data, train_h, val_h, run_dir):
    iec_pred, iec_inp = iec_data["predictions"], iec_data["inputs"]
    mvc_pred, mvc_inp = mvc_data["predictions"], mvc_data["inputs"]
    
    val_losses = [d.get("total", float("inf")) for d in val_h]
    train_losses = [d.get("total", float("inf")) for d in train_h]
    best_epoch = int(np.argmin(val_losses)) + 1
    
    targets = {
        "T3  (IEC dry-ch outlet)": (t(iec_inp["T3"]), t(iec_pred["hat_T3"]), "°C"),
        "Tr1 (IEC wet-ch outlet)": (t(iec_inp["Tr1"]), t(iec_pred["hat_Tr1"]), "°C"),
        "Qiec (IEC cooling capacity)": (t(iec_inp["Qiec"]), t(iec_pred["hat_Qiec"]), "kW"),
        "RH3 (IEC dry-ch outlet RH)": (t(iec_inp["RH3"]), t(iec_pred["hat_RH3"]), "%"),
        "T1  (MVC supply air)": (t(mvc_inp["T1"]), t(mvc_pred["hat_T1"]), "°C"),
        "Qcoil (MVC coil load)": (t(mvc_inp["Qcoil"]), t(mvc_pred["hat_Qcoil"]), "kW"),
        "ECompPower": (t(mvc_inp["ECompPower"]), t(mvc_pred["hat_ECompPower"]), "kW"),
    }
    
    lines = [
        "=" * 68,
        "  EP-PINN — NIEC+MVC System  |  Test Set Results",
        "=" * 68,
        "",
        f"Run directory : {run_dir}",
        f"Best epoch    : {best_epoch}  (early-stopped at {len(val_losses)} / 600 max)",
        f"Best val loss : {min(val_losses):.4f}  (MSE, mixed units)",
        f"Final train   : {train_losses[best_epoch-1]:.4f}",
        "",
        "-" * 68,
        "  Test-set metrics  (model loaded: model_best.pt)",
        "-" * 68,
        f"  {'Target':<28}  {'RMSE':>8}  {'MAE':>8}  {'Bias':>8}  {'R²':>6}  Unit",
    ]
    
    for name, (y_true, y_pred, unit) in targets.items():
        r = rmse(y_true, y_pred)
        m = mae(y_true, y_pred)
        b = bias(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        lines.append(f"  {name:<28}  {r:>8.4f}  {m:>8.4f}  {b:>8.4f}  {r2:>6.3f}  {unit}")
    
    eps_dry_t, eps_dry_p = t(iec_pred["eps_dry_obs"]), t(iec_pred["hat_eps_dry"])
    eps_wet_t, eps_wet_p = t(iec_pred["eps_wet_obs"]), t(iec_pred["hat_eps_wet"])
    
    valid_mask = (eps_wet_t >= 0) & (eps_wet_t <= 1)
    ew_true_val = eps_wet_t[valid_mask]
    ew_pred_val = eps_wet_p[valid_mask]
    
    lines += [
        "",
        "  -- Effectiveness --",
        f"  eps_dry (all rows)              RMSE={rmse(eps_dry_t, eps_dry_p):.4f}  R²={r2_score(eps_dry_t, eps_dry_p):.3f}",
        f"  eps_wet (all rows)              RMSE={rmse(eps_wet_t, eps_wet_p):.4f}  R²={r2_score(eps_wet_t, eps_wet_p):.3f}",
        f"  eps_wet (valid rows only)       RMSE={rmse(ew_true_val, ew_pred_val):.4f}  R²={r2_score(ew_true_val, ew_pred_val):.3f}",
        "=" * 68,
    ]
    
    text = "\n".join(lines)
    with open(os.path.join(run_dir, "results.txt"), "w") as f:
        f.write(text + "\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main(run_dir, show):
    print(f"Visualising: {run_dir}\n")

    train_h, val_h = load_history(run_dir)
    iec_data = load_predictions(run_dir, "iec")
    mvc_data = load_predictions(run_dir, "mvc")

    iec_pred = iec_data["predictions"]
    iec_inp  = iec_data["inputs"]
    mvc_pred = mvc_data["predictions"]
    mvc_inp  = mvc_data["inputs"]

    print("Plotting loss curves...")
    plot_loss_curves(train_h, val_h, run_dir, show)

    # Parity plots — IEC targets
    print("Plotting parity plots...")
    plot_parity(t(iec_inp["T3"]),  t(iec_pred["hat_T3"]),  "T3",  "°C",  run_dir, show, "parity_T3.png")
    plot_parity(t(iec_inp["Tr1"]), t(iec_pred["hat_Tr1"]), "Tr1", "°C",  run_dir, show, "parity_Tr1.png")
    plot_parity(t(iec_inp["Qiec"]),t(iec_pred["hat_Qiec"]),"Qiec","kW",  run_dir, show, "parity_Qiec.png")
    # MVC targets
    plot_parity(t(mvc_inp["T1"]),         t(mvc_pred["hat_T1"]),         "T1",         "°C", run_dir, show, "parity_T1.png")
    plot_parity(t(mvc_inp["Qcoil"]),      t(mvc_pred["hat_Qcoil"]),      "Qcoil",      "kW", run_dir, show, "parity_Qcoil.png")
    plot_parity(t(mvc_inp["ECompPower"]), t(mvc_pred["hat_ECompPower"]), "ECompPower", "kW", run_dir, show, "parity_ECompPower.png")

    print("Plotting effectiveness...")
    plot_effectiveness(iec_data, run_dir, show)

    # Combined targets dict for residual / boxplot / timeseries
    targets = {
        "T3":         (t(iec_inp["T3"]),         t(iec_pred["hat_T3"]),         "°C"),
        "Tr1":        (t(iec_inp["Tr1"]),         t(iec_pred["hat_Tr1"]),        "°C"),
        "Qiec":       (t(iec_inp["Qiec"]),        t(iec_pred["hat_Qiec"]),       "kW"),
        "T1":         (t(mvc_inp["T1"]),          t(mvc_pred["hat_T1"]),         "°C"),
        "Qcoil":      (t(mvc_inp["Qcoil"]),       t(mvc_pred["hat_Qcoil"]),      "kW"),
        "ECompPower": (t(mvc_inp["ECompPower"]),  t(mvc_pred["hat_ECompPower"]), "kW"),
    }

    print("Plotting residuals...")
    plot_residuals(targets, run_dir, show)

    print("Plotting error boxplot...")
    plot_error_boxplot(targets, run_dir, show)

    print("Plotting COP distribution...")
    plot_cop_distribution(mvc_data, run_dir, show)

    print("Plotting time series...")
    plot_timeseries(targets, run_dir, show)

    print("Generating results.txt metrics...")
    save_metrics_txt(iec_data, mvc_data, train_h, val_h, run_dir)

    print(f"\nAll plots and results.txt saved to: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to the Logs-* run directory")
    parser.add_argument("--show", action="store_true",
                        help="Also open interactive plot windows")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg")

    main(args.run_dir, args.show)
