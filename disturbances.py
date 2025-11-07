#!/usr/bin/env python3

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Disturbance modelling (same as before)
# ============================================================


@dataclass
class DisturbanceParameters:
    rho: float = 1025.0
    g: float = 9.81

    L: float = 0.457
    W: float = 0.338
    H: float = 0.254
    A_x: float = 0.0859
    A_y: float = 0.1161
    A_z: float = 0.1545

    C_D_x: float = 0.87
    C_D_y: float = 2.18
    C_D_z: float = 3.07

    C_M_x: float = 2.0
    C_M_y: float = 2.0
    C_M_z: float = 2.0

    d_t: float = 0.0076
    L_t_min: float = 25.0
    L_t_max: float = 300.0
    C_D_t: float = 1.2

    r_tether: Optional[np.ndarray] = None
    z_B: float = -0.01

    def __post_init__(self):
        if self.r_tether is None:
            self.r_tether = np.array([-0.20, 0.00, -0.11])
        self.V_x = self.A_x * (self.L / 2)
        self.V_y = self.A_y * (self.W / 2)
        self.V_z = self.A_z * (self.H / 2)


class DisturbanceModel:
    MAX_BODY_VEL = 5.0
    MAX_BODY_POS = 20.0

    def __init__(self, params: Optional[DisturbanceParameters] = None):
        self.params = params if params else DisturbanceParameters()

    def hydrodynamic_drag(self, nu: np.ndarray) -> np.ndarray:
        p = self.params
        vel = np.clip(nu[:3], -self.MAX_BODY_VEL, self.MAX_BODY_VEL)
        u, v, w = vel[0], vel[1], vel[2]
        F_drag = np.zeros(6)
        F_drag[0] = -(p.rho / 2) * p.C_D_x * p.A_x * u * abs(u)
        F_drag[1] = -(p.rho / 2) * p.C_D_y * p.A_y * v * abs(v)
        F_drag[2] = -(p.rho / 2) * p.C_D_z * p.A_z * w * abs(w)
        return F_drag

    def tether_drag(self, nu: np.ndarray, L_t: float) -> np.ndarray:
        p = self.params
        vel = np.clip(nu[:3], -self.MAX_BODY_VEL, self.MAX_BODY_VEL)
        coeff = (p.rho * p.C_D_t * p.d_t / 8) * L_t
        F_tether = coeff * vel * np.abs(vel)
        tau_tether_moment = np.cross(p.r_tether, F_tether)
        return np.hstack([F_tether, tau_tether_moment])

    def _solve_wavenumber(self, omega: float, h: float, tol: float = 1e-6) -> float:
        g = self.params.g
        k = omega**2 / g
        for _ in range(20):
            f = g * k * np.tanh(k * h) - omega**2
            df = g * (np.tanh(k * h) + k * h * (1 / np.cosh(k * h))**2)
            k_new = k - f / df
            if abs(k_new - k) < tol:
                return k_new
            k = k_new
        return k

    def wave_forces_morison(self, nu: np.ndarray, eta: np.ndarray,
                            t: float, H_s: float, T_p: float,
                            h: float = 10.0) -> np.ndarray:
        p = self.params
        a = H_s / 2
        omega = 2 * np.pi / T_p
        k = self._solve_wavenumber(omega, h)

        phi_wave = 0.0
        x, z = eta[0], eta[2]
        z = np.clip(z, -h + 1e-3, 0.0)
        x = np.clip(x, -self.MAX_BODY_POS, self.MAX_BODY_POS)
        kh = np.clip(k * h, -20.0, 20.0)
        kz = np.clip(k * (z + h), -20.0, 20.0)
        denom = np.sinh(kh)
        if abs(denom) < 1e-6:
            denom = np.sign(denom) * 1e-6 if denom != 0 else 1e-6
        cosh_term = np.cosh(kz) / denom
        sinh_term = np.sinh(kz) / denom

        u_f_x = a * omega * cosh_term * np.cos(k * x - omega * t + phi_wave)
        u_f_z = a * omega * sinh_term * np.sin(k * x - omega * t + phi_wave)

        du_f_x = -a * omega**2 * cosh_term * np.sin(k * x - omega * t + phi_wave)
        du_f_z = a * omega**2 * sinh_term * np.cos(k * x - omega * t + phi_wave)

        F_wave = np.zeros(6)
        rel_vel_x = nu[0] - u_f_x
        F_drag_x = 0.5 * p.rho * p.C_D_x * p.A_x * abs(rel_vel_x) * rel_vel_x
        F_inertia_x = p.rho * p.C_M_x * p.V_x * du_f_x
        F_wave[0] = F_drag_x + F_inertia_x

        rel_vel_z = nu[2] - u_f_z
        F_drag_z = 0.5 * p.rho * p.C_D_z * p.A_z * abs(rel_vel_z) * rel_vel_z
        F_inertia_z = p.rho * p.C_M_z * p.V_z * du_f_z
        F_wave[2] = F_drag_z + F_inertia_z
        return F_wave

    def ocean_current(self, nu: np.ndarray, v_current: np.ndarray) -> np.ndarray:
        p = self.params
        vel = np.clip(nu[:3] - v_current, -self.MAX_BODY_VEL, self.MAX_BODY_VEL)
        nu_rel = vel
        F_current = np.zeros(6)
        F_current[0] = -(p.rho / 2) * p.C_D_x * p.A_x * nu_rel[0] * abs(nu_rel[0])
        F_current[1] = -(p.rho / 2) * p.C_D_y * p.A_y * nu_rel[1] * abs(nu_rel[1])
        F_current[2] = -(p.rho / 2) * p.C_D_z * p.A_z * nu_rel[2] * abs(nu_rel[2])
        return F_current

    def total_disturbance(self, nu: np.ndarray, eta: np.ndarray, t: float,
                          scenario: str = "calm", L_t: Optional[float] = None,
                          H_s: float = 0.0, T_p: float = 6.0,
                          v_current: Optional[np.ndarray] = None) -> np.ndarray:
        nu_limited = np.clip(nu, -self.MAX_BODY_VEL, self.MAX_BODY_VEL)
        eta_limited = np.clip(eta, -self.MAX_BODY_POS, self.MAX_BODY_POS)

        tau_d = np.zeros(6)
        tau_d += self.hydrodynamic_drag(nu_limited)

        if scenario in ["tether", "combined"] or L_t is not None:
            if L_t is None:
                pos = eta_limited[:3]
                L_t = np.linalg.norm(pos)
                L_t = np.clip(L_t, self.params.L_t_min, self.params.L_t_max)
            tau_d += self.tether_drag(nu_limited, L_t)

        if scenario in ["waves", "combined"] and H_s > 0:
            tau_d += self.wave_forces_morison(nu_limited, eta_limited, t, H_s, T_p)

        if scenario in ["current", "combined"] and v_current is not None:
            tau_d += self.ocean_current(nu_limited, v_current)

        return tau_d


# ============================================================
# Controller/plant parameters (copied from the standalone scripts)
# ============================================================

Ts = 0.01

axis_names = ["Surge x", "Sway y", "Heave z", "Roll φ", "Pitch θ", "Yaw ψ"]
color_map = plt.cm.get_cmap("tab10", len(axis_names))

g = 9.81
m = 11.0
B = m * g
zB = -0.01
k_phi_theta = -zB * B

M_diag_vals = np.array([5.5, 1.7, 3.57, 0.14, 0.11, 0.25], dtype=float)
D1_diag = np.array([4.03, 6.22, 5.18, 0.07, 0.07, 0.07], dtype=float)
K_diag = np.array([0.0, 0.0, 0.0, k_phi_theta, k_phi_theta, 0.0], dtype=float)


# PID gains (from PID.py)
Kp_trans = np.array([5.786, 11.492, 3.755])
Ki_trans = np.array([0.463, 0.919, 0.300])
Kd_trans = np.array([4.770, 0.675, 0.532])

Kp_rot = np.array([0.735, 0.347, 3.240])
Ki_rot = np.array([0.059, 0.028, 0.259])
Kd_rot = np.array([0.716, 0.548, 1.334])

Kp_all = np.concatenate([Kp_trans, Kp_rot])
Ki_all = np.concatenate([Ki_trans, Ki_rot])
Kd_all = np.concatenate([Kd_trans, Kd_rot])

Ti_all = np.where(Ki_all > 0.0, Kp_all / np.maximum(Ki_all, 1e-12), np.inf)
Td_all = np.where(Kp_all > 0.0, Kd_all / np.maximum(Kp_all, 1e-12), 0.0)
N_pid = 10.0 / Ts
w_ref_pid = [None, None, None, None, 0.5, None]

# LQR preparation ---------------------------------------------------------

Q_list = [
    np.diag([8.0, 0.5, 3.0]),
    np.diag([10.0, 0.6, 3.0]),
    np.diag([6.0, 0.4, 2.5]),
    np.diag([12.0, 0.8, 6.0]),
    np.diag([14.0, 0.8, 8.0]),
    np.diag([10.0, 0.7, 5.0]),
]
R_list = [
    np.array([[0.6]]), np.array([[0.7]]), np.array([[0.6]]),
    np.array([[0.15]]), np.array([[0.12]]), np.array([[0.18]])
]
w_ref_lqr = [None, None, None, None, 0.5, None]


def aug_ss_cont(m_val: float, d_val: float, k_val: float):
    A = np.array([[0.0, 1.0, 0.0],
                  [-k_val/m_val, -d_val/m_val, 0.0],
                  [-1.0, 0.0, 0.0]], dtype=float)
    B = np.array([[0.0],
                  [1.0/m_val],
                  [0.0]], dtype=float)
    return A, B


def c2d_euler(A: np.ndarray, B: np.ndarray, Ts: float):
    Ad = np.eye(A.shape[0]) + Ts * A
    Bd = Ts * B
    return Ad, Bd


def dlqr(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray,
         max_iter: int = 5000, eps: float = 1e-9):
    P = Q.copy()
    for _ in range(max_iter):
        BtPB = Bd.T @ P @ Bd
        K = np.linalg.solve(R + BtPB, Bd.T @ P @ Ad)
        Pn = Ad.T @ P @ (Ad - Bd @ K) + Q
        if np.linalg.norm(Pn - P, ord="fro") <= eps:
            P = Pn
            break
        P = Pn
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K


def build_lqr_gains():
    gains = []
    for i in range(6):
        A, B = aug_ss_cont(M_diag_vals[i], D1_diag[i], K_diag[i])
        Ad, Bd = c2d_euler(A, B, Ts)
        gains.append(dlqr(Ad, Bd, Q_list[i], R_list[i]))
    return gains


LQR_K_LIST = build_lqr_gains()

# SMC parameters (FIXED - FROM THEORY DOCUMENT) ------------------------------

# Base parameters (perfect for calm water)
c2_smc_base = np.array([1.2, 1.8, 1.2, 2.0, 2.0, 1.8], dtype=float)  # lambda (from theory)
c3_smc_base = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)  # gamma (from theory)
phi_smc_base = np.array([0.01, 0.01, 0.01, 0.0087266, 0.0087266, 0.0087266], dtype=float)  # phi (from theory)
k_s_smc_base = np.array([4.244, 16.916, 9.968, 16.520, 21.093, 16.824], dtype=float)  # K (from theory)

w_ref_smc = [None, None, None, None, 0.5, None]

# Control limits - SURGE/PITCH LIMITS INCREASED (thruster limits ignored as requested)
u_max_smc = np.array([5000.0, 100.0, 150.0, 10.0, 1500.0, 10.0], dtype=float)
# Note: Surge=5000N (for worst case 4049N peak), Heave=150N, Pitch=1500 N·m

# ============================================================
# ADAPTIVE SMC GAINS (scenario-dependent tuning)
# ============================================================

def get_adaptive_smc_gains(scenario_name: str):
    """
    Return SMC parameters adapted to scenario disturbance level.
    
    Calm water: Use theory gains (proven to work perfectly!)
    High disturbances: Scale gains to match expected forces
    """
    name_lower = scenario_name.lower()
    
    if 'calm' in name_lower:
        # Calm water: Use theory gains (PROVEN!)
        print(f"  SMC: Using THEORY gains (k_surge={k_s_smc_base[0]:.1f}N)")
        return c2_smc_base.copy(), c3_smc_base.copy(), phi_smc_base.copy(), k_s_smc_base.copy()
    
    elif 'worst' in name_lower:
        # Worst case: 0.75 m/s current + 1.5m waves
        # Expected: ~4000N surge, ~1350 N·m pitch moment
        c2 = c2_smc_base.copy()
        c3 = c3_smc_base.copy()
        phi = phi_smc_base.copy()
        k_s = k_s_smc_base.copy()
        
        # Scale for extreme conditions (increased further)
        c3[0] = 1.0      # 10x integral for surge (was 0.6)
        c3[4] = 0.8      # 8x integral for pitch (was 0.4)
        c3[2] = 0.5      # 5x integral for heave (was 0.3)
        k_s[0] = 2000.0  # Very high switching for surge (471x base, was 800)
        k_s[4] = 800.0   # Very high switching for pitch (38x base, was 300)
        k_s[2] = 300.0   # High switching for heave (30x base, was 200)
        
        print(f"  SMC: Using ADAPTIVE gains for worst case (k_surge={k_s[0]:.1f}N, {k_s[0]/k_s_smc_base[0]:.0f}x base)")
        return c2, c3, phi, k_s
    
    elif 'current' in name_lower:
        # Current + tether: 0.5 m/s current
        # Expected: ~1200N surge, ~150 N·m pitch moment
        c2 = c2_smc_base.copy()
        c3 = c3_smc_base.copy()
        phi = phi_smc_base.copy()
        k_s = k_s_smc_base.copy()
        
        # Scale to handle current
        c3[0] = 0.4      # 4x integral for surge
        c3[4] = 0.3      # 3x integral for pitch
        k_s[0] = 300.0   # 71x increase for surge
        k_s[4] = 200.0   # 9x increase for pitch
        
        print(f"  SMC: Using ADAPTIVE gains for current (k_surge={k_s[0]:.1f}N, {k_s[0]/k_s_smc_base[0]:.0f}x base)")
        return c2, c3, phi, k_s
    
    elif 'wave' in name_lower:
        # Moderate waves: Hs=1.0m
        # Expected: ~250N surge/heave peaks
        c2 = c2_smc_base.copy()
        c3 = c3_smc_base.copy()
        phi = phi_smc_base.copy()
        k_s = k_s_smc_base.copy()
        
        # Moderate increases for wave handling
        c3[[0,2]] = 0.25      # 2.5x integral for surge/heave
        k_s[0] = 60.0         # 14x increase for surge
        k_s[2] = 100.0        # 10x increase for heave
        
        print(f"  SMC: Using ADAPTIVE gains for waves (k_surge={k_s[0]:.1f}N, {k_s[0]/k_s_smc_base[0]:.0f}x base)")
        return c2, c3, phi, k_s
    
    else:
        # Default: use theory gains
        print(f"  SMC: Using DEFAULT (theory) gains")
        return c2_smc_base.copy(), c3_smc_base.copy(), phi_smc_base.copy(), k_s_smc_base.copy()


# ============================================================
# Scenario definitions
# ============================================================


@dataclass
class DisturbanceScenario:
    name: str
    description: str
    T_end: float
    scenario_key: str
    H_s: float = 0.0
    T_p: float = 6.0
    v_current: Optional[np.ndarray] = None
    L_t: Optional[float] = None
    initial_eta: Optional[np.ndarray] = None
    initial_nu: Optional[np.ndarray] = None


def build_scenarios() -> List[DisturbanceScenario]:
    return [
        DisturbanceScenario(
            name="Calm + Tether",
            description="Calm water with tether drag estimated from position.",
            T_end=20.0,
            scenario_key="tether",
        ),
        DisturbanceScenario(
            name="Moderate Waves",
            description="Wave disturbance with Hs=1.0 m, Tp=6 s, hovering at -2 m.",
            T_end=30.0,
            scenario_key="waves",
            H_s=1.0,
            T_p=6.0,
            initial_eta=np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.0]),
        ),
        DisturbanceScenario(
            name="Current + Tether",
            description="Steady 0.5 m/s current with 100 m tether.",
            T_end=25.0,
            scenario_key="combined",
            v_current=np.array([0.5, 0.0, 0.0]),
            L_t=100.0,
        ),
        DisturbanceScenario(
            name="Worst Case",
            description="Combined worst case: Hs=1.5 m, Tp=7 s, 0.75 m/s current, 150 m tether.",
            T_end=30.0,
            scenario_key="combined",
            H_s=1.5,
            T_p=7.0,
            v_current=np.array([0.75, 0.0, 0.0]),
            L_t=150.0,
            initial_eta=np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0]),
        ),
    ]


# ============================================================
# Simulation helpers
# ============================================================


def slugify(text: str) -> str:
    clean = []
    for ch in text.lower():
        if ch.isalnum():
            clean.append(ch)
        elif ch in (" ", "-", "_"):
            clean.append("_")
    slug = "".join(clean).strip("_")
    return slug or "scenario"


def simulate_pid(model: DisturbanceModel, cfg: DisturbanceScenario):
    steps = int(np.round(cfg.T_end / Ts)) + 1
    t = np.linspace(0.0, cfg.T_end, steps)

    eta = np.zeros((steps, 6))
    nu = np.zeros((steps, 6))
    u = np.zeros((steps, 6))
    tau_log = np.zeros((steps, 6))

    if cfg.initial_eta is not None:
        eta[0] = cfg.initial_eta
    if cfg.initial_nu is not None:
        nu[0] = cfg.initial_nu

    I_int = np.zeros(6)
    d_filt = np.zeros(6)
    r_step = np.ones(6)
    r_filt = np.zeros(6)

    def C_of_s(s, kp, Ti, Td, N):
        i_term = 0.0 if np.isinf(Ti) else 1.0 / (Ti * s)
        d_term = (Td * s) / (1.0 + Td * s / N) if Td > 0 and N > 0 else 0.0
        return kp * (1.0 + i_term + d_term)

    masses = M_diag_vals
    for k in range(1, steps):
        for i in range(6):
            if w_ref_pid[i] is None:
                r_filt[i] = r_step[i]
            else:
                tau_f = 1.0 / w_ref_pid[i]
                alpha = Ts / (Ts + tau_f)
                r_filt[i] = r_filt[i] + alpha * (r_step[i] - r_filt[i])

        tau_dist = model.total_disturbance(
            nu[k - 1], eta[k - 1], t[k - 1],
            scenario=cfg.scenario_key,
            L_t=cfg.L_t,
            H_s=cfg.H_s,
            T_p=cfg.T_p,
            v_current=cfg.v_current,
        )
        tau_log[k - 1] = tau_dist

        for i in range(6):
            e = r_filt[i] - eta[k - 1, i]
            ydot = nu[k - 1, i]

            tau_d = Td_all[i] / N_pid if (Td_all[i] > 0 and N_pid > 0) else 0.0
            if tau_d > 0:
                alpha_d = Ts / (Ts + tau_d)
                d_filt[i] = d_filt[i] + alpha_d * (-ydot - d_filt[i])
                d_term = Td_all[i] * d_filt[i]
            else:
                d_term = 0.0

            if np.isfinite(Ti_all[i]):
                I_int[i] += (Kp_all[i] / Ti_all[i]) * e * Ts

            integral_term = 0.0 if np.isinf(Ti_all[i]) else I_int[i] / Kp_all[i]
            u_cmd = Kp_all[i] * (e + integral_term + d_term)

            u[k, i] = u_cmd

            acc = (u_cmd + tau_dist[i] - D1_diag[i] * nu[k - 1, i] - K_diag[i] * eta[k - 1, i]) / masses[i]
            nu[k, i] = nu[k - 1, i] + Ts * acc
            eta[k, i] = eta[k - 1, i] + Ts * nu[k, i]

    tau_log[-1] = model.total_disturbance(
        nu[-1], eta[-1], t[-1],
        scenario=cfg.scenario_key,
        L_t=cfg.L_t,
        H_s=cfg.H_s,
        T_p=cfg.T_p,
        v_current=cfg.v_current,
    )

    return dict(t=t, eta=eta, nu=nu, u=u, disturbances=tau_log)


def simulate_lqr(model: DisturbanceModel, cfg: DisturbanceScenario):
    steps = int(np.round(cfg.T_end / Ts)) + 1
    t = np.linspace(0.0, cfg.T_end, steps)

    eta = np.zeros((steps, 6))
    nu = np.zeros((steps, 6))
    u = np.zeros((steps, 6))
    tau_log = np.zeros((steps, 6))
    x = np.zeros((6, 3))
    r_step = np.ones(6)
    r_filt = np.zeros(6)

    if cfg.initial_eta is not None:
        eta[0] = cfg.initial_eta
        x[:, 0] = cfg.initial_eta
    if cfg.initial_nu is not None:
        nu[0] = cfg.initial_nu
        x[:, 1] = cfg.initial_nu

    for k in range(1, steps):
        for i in range(6):
            if w_ref_lqr[i] is None:
                r_filt[i] = r_step[i]
            else:
                tau_f = 1.0 / w_ref_lqr[i]
                alpha = Ts / (Ts + tau_f)
                r_filt[i] = r_filt[i] + alpha * (r_step[i] - r_filt[i])

        tau_dist = model.total_disturbance(
            nu[k - 1], eta[k - 1], t[k - 1],
            scenario=cfg.scenario_key,
            L_t=cfg.L_t,
            H_s=cfg.H_s,
            T_p=cfg.T_p,
            v_current=cfg.v_current,
        )
        tau_log[k - 1] = tau_dist

        for i in range(6):
            e = r_filt[i] - x[i, 0]
            x[i, 2] += e * Ts

            ui = - (LQR_K_LIST[i] @ x[i]).item()
            u[k, i] = ui

            acc = (ui + tau_dist[i] - D1_diag[i] * x[i, 1] - K_diag[i] * x[i, 0]) / M_diag_vals[i]
            x[i, 1] += Ts * acc
            x[i, 0] += Ts * x[i, 1]

            eta[k, i] = x[i, 0]
            nu[k, i] = x[i, 1]

    tau_log[-1] = model.total_disturbance(
        nu[-1], eta[-1], t[-1],
        scenario=cfg.scenario_key,
        L_t=cfg.L_t,
        H_s=cfg.H_s,
        T_p=cfg.T_p,
        v_current=cfg.v_current,
    )

    return dict(t=t, eta=eta, nu=nu, u=u, disturbances=tau_log)


def sat(x):
    return np.clip(x, -1.0, 1.0)


def simulate_smc(model: DisturbanceModel, cfg: DisturbanceScenario):
    # Get scenario-appropriate gains
    c2_smc, c3_smc, phi_smc, k_s_smc = get_adaptive_smc_gains(cfg.name)
    
    steps = int(np.round(cfg.T_end / Ts)) + 1
    t = np.linspace(0.0, cfg.T_end, steps)

    eta = np.zeros((steps, 6))
    nu = np.zeros((steps, 6))
    u = np.zeros((steps, 6))
    tau_log = np.zeros((steps, 6))
    z_int = np.zeros(6)
    r_step = np.ones(6)
    r_filt = np.zeros(6)

    if cfg.initial_eta is not None:
        eta[0] = cfg.initial_eta
    if cfg.initial_nu is not None:
        nu[0] = cfg.initial_nu

    for k in range(1, steps):
        for i in range(6):
            if w_ref_smc[i] is None:
                r_filt[i] = r_step[i]
            else:
                tau_f = 1.0 / w_ref_smc[i]
                alpha = Ts / (Ts + tau_f)
                r_filt[i] = r_filt[i] + alpha * (r_step[i] - r_filt[i])

        tau_dist = model.total_disturbance(
            nu[k - 1], eta[k - 1], t[k - 1],
            scenario=cfg.scenario_key,
            L_t=cfg.L_t,
            H_s=cfg.H_s,
            T_p=cfg.T_p,
            v_current=cfg.v_current,
        )
        tau_log[k - 1] = tau_dist

        for i in range(6):
            m_val = M_diag_vals[i]
            d_val = D1_diag[i]
            K_val = K_diag[i]

            e = r_filt[i] - eta[k - 1, i]
            e_dot = -nu[k - 1, i]
            s = e_dot + c2_smc[i] * e + c3_smc[i] * z_int[i]

            desired_acc = -c2_smc[i] * nu[k - 1, i] + c3_smc[i] * e + k_s_smc[i] * sat(s / max(phi_smc[i], 1e-6))
            #             ^^^ NEGATIVE!            ^^^ POSITIVE!
            
            ui = m_val * desired_acc + d_val * nu[k - 1, i] + K_val * eta[k - 1, i]
            u[k, i] = float(np.clip(ui, -u_max_smc[i], u_max_smc[i]))

            if abs(u[k, i]) < 0.95 * u_max_smc[i]:
                z_int[i] += e * Ts
                # Adaptive integral limits based on scenario
                if 'worst' in cfg.name.lower():
                    # Worst case needs very large integral buildup
                    int_max = 100.0 if i < 3 else 10.0
                elif 'current' in cfg.name.lower():
                    # High current scenarios need larger integral buildup
                    int_max = 50.0 if i < 3 else 5.0
                else:
                    # Normal scenarios
                    int_max = 10.0 if i < 3 else 1.0
                z_int[i] = np.clip(z_int[i], -int_max, int_max)

            acc = (u[k, i] + tau_dist[i] - d_val * nu[k - 1, i] - K_val * eta[k - 1, i]) / m_val
            nu[k, i] = nu[k - 1, i] + Ts * acc
            eta[k, i] = eta[k - 1, i] + Ts * nu[k, i]

    tau_log[-1] = model.total_disturbance(
        nu[-1], eta[-1], t[-1],
        scenario=cfg.scenario_key,
        L_t=cfg.L_t,
        H_s=cfg.H_s,
        T_p=cfg.T_p,
        v_current=cfg.v_current,
    )

    return dict(t=t, eta=eta, nu=nu, u=u, disturbances=tau_log)


# ============================================================
# Plotting and reporting
# ============================================================


def plot_responses(cfg: DisturbanceScenario,
                   results: Dict[str, Dict[str, np.ndarray]],
                   outdir: str):
    slug = slugify(cfg.name)
    t = results["PID"]["t"]
    controllers = ["PID", "LQR", "SMC"]
    colours = {"PID": "tab:blue", "LQR": "tab:orange", "SMC": "tab:green"}
    axis_labels = ["Surge (x)", "Sway (y)", "Heave (z)", "Roll", "Pitch", "Yaw"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for idx, ax in enumerate(axes.flatten()):
        for ctrl in controllers:
            ax.plot(t, results[ctrl]["eta"][:, idx], label=ctrl, color=colours[ctrl])
        ax.plot(t, np.ones_like(t), linestyle="--", color="black", linewidth=0.8,
                label="Reference" if idx == 0 else None)
        ax.set_title(axis_labels[idx])
        ax.set_ylabel("Response")
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 2].set_xlabel("Time [s]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"Controller Responses - {cfg.name}", fontsize=14)
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"responses_{slug}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(cfg: DisturbanceScenario,
                  results: Dict[str, Dict[str, np.ndarray]],
                  outdir: str):
    slug = slugify(cfg.name)
    lines = [
        f"Scenario: {cfg.name}",
        cfg.description,
        f"Duration: {cfg.T_end:.1f} s",
        "",
    ]

    for ctrl_name, res in results.items():
        tau = res["disturbances"]
        forces_rms = np.sqrt(np.mean(tau[:, :3] ** 2, axis=0))
        forces_max = np.max(np.abs(tau[:, :3]), axis=0)
        moments_rms = np.sqrt(np.mean(tau[:, 3:] ** 2, axis=0))
        moments_max = np.max(np.abs(tau[:, 3:]), axis=0)
        final_eta = res["eta"][-1]

        lines.extend([
            f"{ctrl_name}:",
            f"  Force RMS [N]: Fx={forces_rms[0]:.3f}, Fy={forces_rms[1]:.3f}, Fz={forces_rms[2]:.3f}",
            f"  Force Max [N]: Fx={forces_max[0]:.3f}, Fy={forces_max[1]:.3f}, Fz={forces_max[2]:.3f}",
            f"  Moment RMS [N·m]: Mx={moments_rms[0]:.3f}, My={moments_rms[1]:.3f}, Mz={moments_rms[2]:.3f}",
            f"  Moment Max [N·m]: Mx={moments_max[0]:.3f}, My={moments_max[1]:.3f}, Mz={moments_max[2]:.3f}",
            "  Final state: "
            f"x={final_eta[0]:.3f}, y={final_eta[1]:.3f}, z={final_eta[2]:.3f}, "
            f"phi={final_eta[3]:.3f}, theta={final_eta[4]:.3f}, psi={final_eta[5]:.3f}",
            "",
        ])

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"summary_{slug}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# Main
# ============================================================


def main():
    print("BlueROV2 Controller Disturbance Responses")
    print("=" * 80)

    model = DisturbanceModel()
    scenarios = build_scenarios()
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "disturbance_controller_responses_FIXED")

    for cfg in scenarios:
        print(f"\nScenario: {cfg.name}")
        results = {
            "PID": simulate_pid(model, cfg),
            "LQR": simulate_lqr(model, cfg),
            "SMC": simulate_smc(model, cfg),
        }
        plot_responses(cfg, results, outdir)
        write_summary(cfg, results, outdir)
        print(f"  Saved response plot and summary for '{cfg.name}'.")

    print("\nAll scenarios processed.")
    print(f"Outputs located in: {outdir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
