"""
train.py  —  Train the EP-PINN model.

Two DataLoaders run in parallel each epoch:
  iec_loader  – IEC dataset (all files, Qiec > 0) → supervises SubNet1/2
  mvc_loader  – MVC dataset (file 1, ECompPower > 0) → supervises SubNet3

The IEC loader drives the epoch length; the MVC loader is cycled to match.
All logs, plots, and the model checkpoint are saved to:
  runs/Logs-HH-MM-DD-Mon-YYYY/

Usage:
    python train.py [--epochs N] [--lr LR] [--lam LAM] [--batch B]
"""

import os
import sys
import csv
import json
import time
import argparse
import datetime
import itertools

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model import EPPINN, ep_pinn_loss

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")

DEFAULTS = dict(epochs=600, lr=1e-3, lam=1.0, batch=256,
                hidden_eff=[64, 64, 32], hidden_mvc=[128, 128, 64],
                warmup=30, grad_clip=1.0, weight_decay=1e-5,
                dropout=0.1, early_stop=80)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset helpers ───────────────────────────────────────────────────────────

def make_loader(pt_file, batch, shuffle):
    d = torch.load(pt_file, map_location="cpu")
    keys = sorted(d.keys())
    tensors = tuple(d[k] for k in keys)
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False), keys


def batch_to_dict(batch, keys):
    return {k: v.to(DEVICE) for k, v in zip(keys, batch)}


# ── Forward pass wrappers ─────────────────────────────────────────────────────

def run_forward(model, b):
    """Unpack a batch dict and call model.forward with the right arguments."""
    return model(
        Tr2=b["Tr2"], wr2=b["wr2"],
        T4=b["T4"], RH4=b["RH4"], w4=b["w4"],
        T2=b["T2"], T1_meas=b["T1"],
        CMHoa=b["CMHoa"], CMHsa=b["CMHsa"],
        T3_meas=b["T3"], Tr1_meas=b["Tr1"],
    )


def iec_targets(b):
    return {"T3": b["T3"], "Tr1": b["Tr1"], "RH3": b["RH3"], "Qiec": b["Qiec"]}


def mvc_targets(b):
    return {"T1": b["T1"], "Qcoil": b["Qcoil"], "ECompPower": b["ECompPower"]}


# ── Loss: combined IEC + MVC in one step ─────────────────────────────────────

def combined_loss(model, iec_b, mvc_b, lam):
    """
    IEC batch → SubNet1/2 physics losses + T3/Tr1/RH3/Qiec observation losses.
    MVC batch → SubNet3 observation losses (T1, Qcoil, ECompPower).
    Both share a single backward pass.
    """
    # IEC pass: carries gradients into SubNet1 and SubNet2
    out_iec = run_forward(model, iec_b)
    mse = nn.functional.mse_loss

    L_T3   = mse(out_iec["hat_T3"],      iec_b["T3"])
    L_Tr1  = mse(out_iec["hat_Tr1"],     iec_b["Tr1"])
    L_RH3  = lam * mse(out_iec["hat_RH3"],  iec_b["RH3"])
    L_Qiec = lam * mse(out_iec["hat_Qiec"], iec_b["Qiec"])
    L_edry = lam * mse(out_iec["hat_eps_dry"],
                        out_iec["eps_dry_obs"].detach())

    # eps_wet loss only on rows where Tr1 < T4 (physically valid)
    mask = iec_b["eps_wet_valid"]          # [B, 1] float, 0 or 1
    n_valid = mask.sum().clamp(min=1)
    ewet_sq = (out_iec["hat_eps_wet"] - out_iec["eps_wet_obs"].detach()) ** 2
    L_ewet = lam * (ewet_sq * mask).sum() / n_valid

    # MVC pass: carries gradients into SubNet3
    out_mvc = run_forward(model, mvc_b)
    L_T1    = mse(out_mvc["hat_T1"],         mvc_b["T1"])
    L_Qcoil = mse(out_mvc["hat_Qcoil"],      mvc_b["Qcoil"])
    L_Ecomp = mse(out_mvc["hat_ECompPower"], mvc_b["ECompPower"])

    total = L_T3 + L_Tr1 + L_RH3 + L_Qiec + L_edry + L_ewet + L_T1 + L_Qcoil + L_Ecomp

    breakdown = dict(
        T3=L_T3.item(), Tr1=L_Tr1.item(), RH3=L_RH3.item(),
        Qiec=L_Qiec.item(), eps_dry=L_edry.item(), eps_wet=L_ewet.item(),
        T1=L_T1.item(), Qcoil=L_Qcoil.item(), ECompPower=L_Ecomp.item(),
        total=total.item(),
    )
    return total, breakdown


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, iec_loader, mvc_cycle, optimizer, lam, train=True):
    model.train(train)
    ctx = torch.enable_grad if train else torch.no_grad
    sums = {}
    n = 0
    with ctx():
        for iec_batch in iec_loader:
            mvc_batch = next(mvc_cycle)
            iec_b = batch_to_dict(iec_batch, iec_keys)
            mvc_b = batch_to_dict(mvc_batch, mvc_keys)

            loss, bd = combined_loss(model, iec_b, mvc_b, lam)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            for k, v in bd.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1

    return {k: v / n for k, v in sums.items()}


# ── Logging ───────────────────────────────────────────────────────────────────

def init_log(run_dir, fieldnames):
    path = os.path.join(run_dir, "training_log.csv")
    f = open(path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=["epoch", "split"] + fieldnames)
    w.writeheader()
    return f, w


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    global iec_keys, mvc_keys   # used in run_epoch closure

    # Run directory
    ts = datetime.datetime.now().strftime("Logs-%H-%M-%d-%b-%Y")
    run_dir = os.path.join(RUNS_DIR, ts)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"Device: {DEVICE}")

    # Save config
    cfg = vars(args)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Data
    iec_tr_loader, iec_keys = make_loader(
        os.path.join(DATA_DIR, "iec_train.pt"), args.batch, shuffle=True)
    iec_va_loader, _        = make_loader(
        os.path.join(DATA_DIR, "iec_val.pt"),   args.batch, shuffle=False)
    mvc_tr_loader, mvc_keys = make_loader(
        os.path.join(DATA_DIR, "mvc_train.pt"), args.batch, shuffle=True)
    mvc_va_loader, _        = make_loader(
        os.path.join(DATA_DIR, "mvc_val.pt"),   args.batch, shuffle=False)

    # Model
    model = EPPINN(
        hidden_eff=args.hidden_eff,
        hidden_mvc=args.hidden_mvc,
        dropout=args.dropout,
    ).to(DEVICE)

    norm_stats = torch.load(
        os.path.join(DATA_DIR, "norm_stats.pt"), map_location=DEVICE
    )
    model.set_normalisation(norm_stats)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Linear warmup then ReduceLROnPlateau
    def lr_lambda(epoch):
        if epoch < args.warmup:
            return float(epoch + 1) / args.warmup
        return 1.0

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=30, factor=0.5, min_lr=1e-6, verbose=False
    )

    # Log file
    loss_keys = ["T3", "Tr1", "RH3", "Qiec", "eps_dry", "eps_wet",
                 "T1", "Qcoil", "ECompPower", "total"]
    log_file, log_writer = init_log(run_dir, loss_keys)

    best_val   = float("inf")
    best_epoch = 0
    history    = {"train": [], "val": []}

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):

        mvc_tr_cycle = itertools.cycle(mvc_tr_loader)
        mvc_va_cycle = itertools.cycle(mvc_va_loader)

        tr_bd = run_epoch(model, iec_tr_loader, mvc_tr_cycle,
                          optimizer, args.lam, train=True)
        va_bd = run_epoch(model, iec_va_loader, mvc_va_cycle,
                          optimizer, args.lam, train=False)

        if epoch <= args.warmup:
            warmup_sched.step()
        else:
            plateau_sched.step(va_bd["total"])

        # Log to CSV
        for split, bd in [("train", tr_bd), ("val", va_bd)]:
            row = {"epoch": epoch, "split": split}
            row.update({k: bd.get(k, float("nan")) for k in loss_keys})
            log_writer.writerow(row)
        log_file.flush()

        history["train"].append(tr_bd)
        history["val"].append(va_bd)

        # Checkpoint best + early stopping
        if va_bd["total"] < best_val:
            best_val = va_bd["total"]
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(run_dir, "model_best.pt"))

        if epoch - best_epoch >= args.early_stop:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {args.early_stop} epochs).")
            break

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train={tr_bd['total']:.4f}  val={va_bd['total']:.4f}  "
                  f"lr={lr_now:.2e}  [{elapsed:.0f}s]")

    log_file.close()

    # Final checkpoint
    torch.save(model.state_dict(), os.path.join(run_dir, "model_final.pt"))

    # Save full history as JSON for visualisation
    # Convert floats to serialisable
    def clean(d): return {k: float(v) for k, v in d.items()}
    hist_out = {"train": [clean(d) for d in history["train"]],
                "val":   [clean(d) for d in history["val"]]}
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(hist_out, f)

    # Save per-split predictions on test set for visualisation
    print("\nGenerating test predictions...")
    _save_test_predictions(model, run_dir, args)

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"All outputs saved to: {run_dir}")


def _save_test_predictions(model, run_dir, args):
    """Run model on test sets and save predictions + targets as tensors."""
    model.eval()

    for mode, pt_name, keys_ref in [
        ("iec", "iec_test.pt", iec_keys),
        ("mvc", "mvc_test.pt", mvc_keys),
    ]:
        loader, keys = make_loader(
            os.path.join(DATA_DIR, pt_name), batch=512, shuffle=False
        )
        all_out, all_inp = [], []
        with torch.no_grad():
            for batch in loader:
                b = batch_to_dict(batch, keys)
                out = run_forward(model, b)
                all_out.append({k: v.cpu() for k, v in out.items()})
                all_inp.append({k: v.cpu() for k, v in b.items()})

        # Concatenate
        def cat(dicts, key):
            return torch.cat([d[key] for d in dicts], dim=0)

        combined_out = {k: cat(all_out, k) for k in all_out[0]}
        combined_inp = {k: cat(all_inp, k) for k in all_inp[0]}

        torch.save({"predictions": combined_out, "inputs": combined_inp},
                   os.path.join(run_dir, f"{mode}_test_predictions.pt"))
        print(f"  Saved {mode}_test_predictions.pt  ({len(loader.dataset)} samples)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--lr",           type=float, default=DEFAULTS["lr"])
    parser.add_argument("--lam",          type=float, default=DEFAULTS["lam"])
    parser.add_argument("--batch",        type=int,   default=DEFAULTS["batch"])
    parser.add_argument("--warmup",       type=int,   default=DEFAULTS["warmup"])
    parser.add_argument("--grad_clip",    type=float, default=DEFAULTS["grad_clip"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--dropout",      type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--early_stop",   type=int,   default=DEFAULTS["early_stop"])
    parser.add_argument("--hidden_eff",   type=int,   nargs="+",
                        default=DEFAULTS["hidden_eff"])
    parser.add_argument("--hidden_mvc",   type=int,   nargs="+",
                        default=DEFAULTS["hidden_mvc"])
    args = parser.parse_args()
    main(args)
