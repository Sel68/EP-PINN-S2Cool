"""
EP-PINN for NIEC+MVC system.

Forward pass order (equations.tex):
  Raw inputs → Direct layers A1 → SubNet1 (ε_dry) → Direct layers A2
             → SubNet2 (ε_wet) → Direct layer A3 → SubNet3 (MVC)
             → Direct layer output (COP)

SubNet inputs (AFR dropped — zero variance in dataset, CMHoa==CMHsa always):
  SubNet1 : (V_oa, Tr2)
  SubNet2 : (V_oa, T4)
  SubNet3 : (T̂3, T2, T̂r1, CMHsa)

Trainable blocks: SubNet1, SubNet2, SubNet3.
All other nodes are fixed arithmetic — no learnable weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Physical constants ────────────────────────────────────────────────────────
RHO_AIR   = 1.2      # kg/m³ (standard dry air density)
CP_AIR    = 1.006    # kJ/(kg·K)
CP_VAP    = 1.86     # kJ/(kg·K)
H_FG      = 2501.0   # kJ/kg  (latent heat at 0 °C)
P_ATM     = 101.325  # kPa

# Effectiveness output bounds (dimensionless, [0, 1] physical range)
EPS_MIN, EPS_MAX = 0.0, 1.0


# ── Helpers: fixed psychrometric functions ────────────────────────────────────

def enthalpy(T, w):
    """Moist air enthalpy h [kJ/kg_da]. w in g/kg → divide by 1000."""
    return CP_AIR * T + (w / 1000.0) * (H_FG + CP_VAP * T)


def wet_bulb_stull(T, RH):
    """Stull (2011) Twb approximation. T in °C, RH in %."""
    return (T * torch.arctan(0.151977 * (RH + 8.313659).sqrt())
            + torch.arctan(T + RH)
            - torch.arctan(RH - 1.676331)
            + 0.00391838 * RH ** 1.5 * torch.arctan(0.023101 * RH)
            - 4.686035)


def pws(T):
    """Saturation vapour pressure [kPa] (Magnus / WMO form)."""
    return 0.6108 * torch.exp(17.27 * T / (T + 237.3))


def pw(w):
    """Partial vapour pressure [kPa] from humidity ratio w [g/kg]."""
    w_kgkg = w / 1000.0
    return P_ATM * w_kgkg / (0.622 + w_kgkg)


def rh_from_w_T(w, T):
    """RH [%] from humidity ratio [g/kg] and dry-bulb T [°C]."""
    return 100.0 * pw(w) / pws(T)


# ── ANN sub-network factory ───────────────────────────────────────────────────

def _mlp(in_dim, hidden, out_dim, activation=nn.Tanh, batch_norm=True, dropout=0.1):
    """Fully-connected block. BatchNorm + Dropout after each activation."""
    layers = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        if batch_norm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ── EP-PINN ───────────────────────────────────────────────────────────────────

class EPPINN(nn.Module):
    """
    Inputs (all raw, un-normalised; caller passes tensors of shape [B, 1]):
        Tr2, wr2             – outdoor air (dry channel in)
        T4, RH4, w4          – room exhaust (wet channel in)
        T2                   – mixed air (T3/T4 mix point)
        T1_meas              – measured MVC supply air (supervision target)
        CMHoa                – outdoor air volumetric flow [m³/hr]  (= V_oa)
        CMHsa                – supply air volumetric flow [m³/hr]   (= V_sa)
        T3_meas              – measured IEC dry-channel outlet (supervision only)
        Tr1_meas             – measured IEC wet-channel outlet (supervision only)

    Outputs (named dict):
        hat_T3, hat_Tr1      – IEC dry/wet channel outlet temperatures
        hat_RH3              – IEC dry-channel outlet RH
        hat_Qiec             – IEC cooling capacity [kW]
        hat_T1               – MVC supply temperature
        hat_Qcoil            – MVC coil load [kW]
        hat_ECompPower       – Compressor power [kW]
        hat_COP              – System COP (derived, not in loss)
        eps_dry_obs          – ε_dry from data (physics loss target)
        eps_wet_obs          – ε_wet from data (physics loss target)
        hat_eps_dry          – Sub-net 1 output
        hat_eps_wet          – Sub-net 2 output
    """

    def __init__(self,
                 hidden_eff=(64, 64, 32),       # SubNets 1 & 2
                 hidden_mvc=(128, 128, 64),      # SubNet 3
                 dropout=0.1):
        super().__init__()

        # Sub-network 1: ε_dry  — inputs: (V_oa, Tr2)
        self.subnet1 = _mlp(2, hidden_eff, 1, dropout=dropout)

        # Sub-network 2: ε_wet  — inputs: (V_oa, T4)
        self.subnet2 = _mlp(2, hidden_eff, 1, dropout=dropout)

        # Sub-network 3: MVC block — inputs: (T̂3, T2, T̂r1, CMHsa)
        self.subnet3 = _mlp(4, hidden_mvc, 3, dropout=dropout)

        # z-score normalisation buffers (not trained, saved with state_dict)
        for name in ["Voa", "Tr2", "T4", "CMHsa",
                     "T2", "T1", "T3hat", "Tr1hat"]:
            self.register_buffer(f"mu_{name}",  torch.zeros(1))
            self.register_buffer(f"std_{name}", torch.ones(1))

    # -- normalisation helpers ------------------------------------------------

    def _norm(self, x, name):
        mu  = getattr(self, f"mu_{name}")
        std = getattr(self, f"std_{name}")
        return (x - mu) / (std + 1e-8)

    def set_normalisation(self, stats: dict):
        """
        Call once before training with a dict of {name: (mean, std)} tensors.
        Expected keys: Voa, Tr2, AFR, T4, CMHsa, T2, T1, T3hat, Tr1hat.
        """
        for name, (mu, std) in stats.items():
            getattr(self, f"mu_{name}").copy_(mu)
            getattr(self, f"std_{name}").copy_(std)

    # -- effectiveness bounded output -----------------------------------------

    @staticmethod
    def _bound_eps(logit):
        """Sigmoid-scale to [EPS_MIN, EPS_MAX]."""
        return torch.sigmoid(logit) * (EPS_MAX - EPS_MIN) + EPS_MIN

    # -- forward pass ---------------------------------------------------------

    def forward(self, Tr2, wr2, T4, RH4, w4, T2, T1_meas,
                CMHoa, CMHsa, T3_meas, Tr1_meas):
        """
        All tensor inputs: shape [B, 1], dtype float32.
        """

        # ── Direct block A1: inlet properties ────────────────────────────────
        h_r2      = enthalpy(Tr2, wr2)
        h_4       = enthalpy(T4, w4)           # noqa: F841
        T_wb4     = wet_bulb_stull(T4, RH4)
        m_dot_oa  = CMHoa * RHO_AIR / 3600.0   # kg/s
        # AFR = CMHsa/CMHoa = 1.0 always in this dataset — not passed to subnets

        # ε_dry from data — physics-loss target only
        eps_dry_obs = (Tr2 - T3_meas) / (Tr2 - T_wb4 + 1e-8)

        # ── Sub-network 1: ε_dry  — inputs: (V_oa, Tr2) ───────────────────────
        sn1_in = torch.cat([
            self._norm(CMHoa, "Voa"),
            self._norm(Tr2,   "Tr2"),
        ], dim=-1)
        hat_eps_dry = self._bound_eps(self.subnet1(sn1_in))

        # ── Direct block A2: dry-channel outlet ──────────────────────────────
        hat_T3   = Tr2 - hat_eps_dry * (Tr2 - T_wb4)
        hat_w3   = wr2.clone()                 # dry channel adds no moisture
        hat_RH3  = rh_from_w_T(hat_w3, hat_T3)
        hat_h3   = enthalpy(hat_T3, hat_w3)
        hat_Qiec = m_dot_oa * (h_r2 - hat_h3)  # kW

        # ε_wet from data — physics-loss target only
        eps_wet_obs = (T4 - Tr1_meas) / (T4 - T_wb4 + 1e-8)

        # ── Sub-network 2: ε_wet  — inputs: (V_oa, T4) ──────────────────────
        sn2_in = torch.cat([
            self._norm(CMHoa, "Voa"),
            self._norm(T4,    "T4"),
        ], dim=-1)
        hat_eps_wet = self._bound_eps(self.subnet2(sn2_in))

        # ── Direct block A3: wet-channel outlet ──────────────────────────────
        hat_Tr1 = T4 - hat_eps_wet * (T4 - T_wb4)

        # ── Sub-network 3: MVC  — inputs: (T̂3, T2, T̂r1, CMHsa) ──────────
        mvc_in  = torch.cat([
            self._norm(hat_T3,   "T3hat"),
            self._norm(T2,       "T2"),
            self._norm(hat_Tr1,  "Tr1hat"),
            self._norm(CMHsa,    "CMHsa"),
        ], dim=-1)
        mvc_out = self.subnet3(mvc_in)         # [B, 3]

        hat_T1         = mvc_out[:, 0:1]
        hat_Qcoil      = mvc_out[:, 1:2]
        hat_ECompPower = mvc_out[:, 2:3]

        # ── Direct output: system COP (derived, not in loss) ──────────────────
        hat_COP = (hat_Qiec + hat_Qcoil) / (hat_ECompPower + 1e-8)

        return dict(
            hat_T3          = hat_T3,
            hat_Tr1         = hat_Tr1,
            hat_RH3         = hat_RH3,
            hat_Qiec        = hat_Qiec,
            hat_T1          = hat_T1,
            hat_Qcoil       = hat_Qcoil,
            hat_ECompPower  = hat_ECompPower,
            hat_COP         = hat_COP,
            eps_dry_obs     = eps_dry_obs,
            eps_wet_obs     = eps_wet_obs,
            hat_eps_dry     = hat_eps_dry,
            hat_eps_wet     = hat_eps_wet,
        )


# ── Loss function ─────────────────────────────────────────────────────────────

def ep_pinn_loss(out: dict, targets: dict, lam: float = 1.0):
    """
    targets keys: T3, Tr1, T1, Qcoil, ECompPower, Qiec, RH3
    lam: physics-loss weight λ (default 1 per DESIGN.md §8).

    Returns total loss and a breakdown dict for logging.
    """
    mse = F.mse_loss

    # Observation losses
    L_obs = (mse(out["hat_T3"],         targets["T3"])
           + mse(out["hat_Tr1"],        targets["Tr1"])
           + mse(out["hat_T1"],         targets["T1"])
           + mse(out["hat_Qcoil"],      targets["Qcoil"])
           + mse(out["hat_ECompPower"], targets["ECompPower"]))

    # Physics-consistency losses
    L_eps_dry = lam * mse(out["hat_eps_dry"], out["eps_dry_obs"].detach())
    L_eps_wet = lam * mse(out["hat_eps_wet"], out["eps_wet_obs"].detach())
    L_Qiec    = lam * mse(out["hat_Qiec"],    targets["Qiec"])
    L_RH3     = lam * mse(out["hat_RH3"],     targets["RH3"])

    total = L_obs + L_eps_dry + L_eps_wet + L_Qiec + L_RH3

    breakdown = dict(
        obs=L_obs.item(),
        eps_dry=L_eps_dry.item(),
        eps_wet=L_eps_wet.item(),
        Qiec=L_Qiec.item(),
        RH3=L_RH3.item(),
        total=total.item(),
    )
    return total, breakdown
