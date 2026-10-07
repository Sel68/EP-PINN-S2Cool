# EP-PINN for NIEC+MVC System

Reference Paper: [*Embedding physical neurons in physics-informed neural networks (EP-PINNs) for enhancing chiller performance prediction*](https://link.springer.com/article/10.1007/s12273-025-1293-z).

---

##  Requirements : 
Python 3 with 


* `torch`
* `numpy`
* `pandas`
* `matplotlib`

---

## How to Run

### 1. Data
Put data in data/ folder. Use ```converter.py``` to get .csv incase of .xlsx 

### 2. Preprocessing the Data
Run
```bash
python3 preprocessing.py
```
*Outputs are saved to `data/processed/`.*

This script removes physically invalid rows, performs random 70/15/15 chronological splits, and gets standardisation metrics.

### 3. Training the Model
Run the main training loop. The script uses early stopping (80 epochs patience) and a LR plateau scheduler. See optional/default arguments for more details.

```bash
python3 train.py
```

**Optional Arguments:**
* `--epochs` (default: 600)
* `--batch` (default: 256)
* `--lr` (default: 1e-3)
* `--early_stop` (default: 80)
* `--dropout` (default: 0.1)

*All models, config details, and predictions on the test set are saved to a timestamped folder inside `runs/` (e.g., `runs/Logs-10-55-03-Oct-2026/`).*

### 3. Visualising Results
Get Parity plots, Loss curves, Residuals, etc. and calculate exact metrics (RMSE, MAE, R²) on the test set. Point the visualisation script to a completed run directory.

```bash
python3 visualise.py runs/Logs-<timestamp>/
```

*Plots are saved as PNG files inside the target run directory.*
