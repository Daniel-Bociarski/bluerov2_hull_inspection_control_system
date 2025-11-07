#!/usr/bin/env python3
# PID closed-loop simulation and frequency analysis for BlueROV2 axes

import numpy as np
import matplotlib.pyplot as plt
import os

# ----------- helpers -----------
def linear_interpolate(x0, x1, y0, y1, target):
    if y1 == y0:
        return x0
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)

def compute_margins(w, L):
    mag = np.abs(L)
    mag_db = 20.0 * np.log10(mag)
    phase_rad = np.unwrap(np.angle(L))
    phase_deg = np.degrees(phase_rad)

    gain_crossover_freq = None
    phase_margin = None
    gain_margin = None
    phase_crossover_freq = None

    # gain crossover: |L| = 1 (0 dB)
    for idx in range(len(w) - 1):
        if (mag_db[idx] - 0.0) * (mag_db[idx + 1] - 0.0) <= 0.0:
            gain_crossover_freq = linear_interpolate(
                w[idx], w[idx + 1], mag_db[idx], mag_db[idx + 1], 0.0
            )
            phase_at_gc_deg = np.degrees(
                np.interp(gain_crossover_freq, w, phase_rad)
            )
            # bring phase close to [-180, 180] for margin computation
            while phase_at_gc_deg <= -180.0:
                phase_at_gc_deg += 360.0
            while phase_at_gc_deg > 180.0:
                phase_at_gc_deg -= 360.0
            phase_margin = 180.0 + phase_at_gc_deg
            break

    # phase crossover: phase = -180 deg
    for idx in range(len(w) - 1):
        if (phase_deg[idx] + 180.0) * (phase_deg[idx + 1] + 180.0) <= 0.0:
            phase_crossover_freq = linear_interpolate(
                w[idx], w[idx + 1], phase_deg[idx], phase_deg[idx + 1], -180.0
            )
            mag_at_pc_db = np.interp(
                phase_crossover_freq, w, mag_db
            )
            gain_margin = -mag_at_pc_db  # in dB
            break

    return {
        "gain_crossover_freq": gain_crossover_freq,
        "phase_margin": phase_margin,
        "phase_crossover_freq": phase_crossover_freq,
        "gain_margin_db": gain_margin,
    }

def compute_step_characteristics(t, response):
    steady_state = response[-1]
    abs_final = max(abs(steady_state), 1e-6)
    overshoot = (np.max(response) - steady_state) / abs_final * 100.0
    tolerance = 0.02 * abs_final
    error = np.abs(response - steady_state)

    settling_time = None
    for idx in range(len(t)):
        if np.all(error[idx:] <= tolerance):
            settling_time = t[idx]
            break

    target_low = steady_state * 0.1
    target_high = steady_state * 0.9
    rise_time = None
    low_cross = None
    high_cross = None
    for idx in range(1, len(t)):
        if low_cross is None:
            if (response[idx - 1] - target_low) * (response[idx] - target_low) <= 0:
                low_cross = linear_interpolate(
                    t[idx - 1], t[idx], response[idx - 1], response[idx], target_low
                )
        if high_cross is None:
            if (response[idx - 1] - target_high) * (response[idx] - target_high) <= 0:
                high_cross = linear_interpolate(
                    t[idx - 1], t[idx], response[idx - 1], response[idx], target_high
                )
        if low_cross is not None and high_cross is not None:
            break
    if low_cross is not None and high_cross is not None:
        rise_time = max(high_cross - low_cross, 0.0)

    damping_ratio = None
    if overshoot > 0.0:
        ln_os = np.log(overshoot / 100.0)
        damping_ratio = -ln_os / np.sqrt(np.pi**2 + ln_os**2)

    return {
        "steady_state": steady_state,
        "overshoot": overshoot,
        "settling_time": settling_time,
        "rise_time": rise_time,
        "damping_ratio": damping_ratio,
    }

def ascii_label(label):
    replacements = {
        "φ": "phi",
        "θ": "theta",
        "ψ": "psi",
    }
    for key, value in replacements.items():
        label = label.replace(key, value)
    return label

# ----------- output folder -----------
script_dir = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(script_dir, "pid_analysis")
os.makedirs(outdir, exist_ok=True)

# ----------- plant parameters -----------
g = 9.81
m = 11.0
B = m * g
zB = -0.01
k_phi_theta = -zB * B  # ≈ 1.0791

# M from doc; use absolute masses/inertias for physics
M_diag = np.array([5.5, -1.7, -3.57, 0.14, 0.11, 0.25], dtype=float)
D1_diag = np.array([4.03, 6.22, 5.18, 0.07, 0.07, 0.07], dtype=float)
masses = np.abs(M_diag)
damps  = D1_diag.copy()
stiff  = np.array([0.0, 0.0, 0.0, k_phi_theta, k_phi_theta, 0.0], dtype=float)

# ----------- PID gains (from tables) -----------
Kp_trans = np.array([5.786, 11.492, 3.755])
Ki_trans = np.array([0.463, 0.919, 0.300])
Kd_trans = np.array([4.770, 0.675, 0.532])
Ti_trans = Kp_trans / Ki_trans
Td_trans = Kd_trans / Kp_trans

Kp_rot = np.array([5.500, 4.000, 3.240])
Ki_rot = np.array([1.400, 4.000, 0.259])
Kd_rot = np.array([1.500, 0.800, 1.334])
Ti_rot = Kp_rot / Ki_rot
Td_rot = Kd_rot / Kp_rot

Kp_all = np.concatenate([Kp_trans, Kp_rot])
Ki_all = np.concatenate([Ki_trans, Ki_rot])
Ti_all = np.concatenate([Ti_trans, Ti_rot])
Kd_all = np.concatenate([Kd_trans, Kd_rot])
Td_all = np.concatenate([Td_trans, Td_rot])

# derivative filter parameter: N = 10/Ts (given)
Ts = 0.01
N  = 10.0 / Ts

# ----------- frequency-domain helpers -----------
def C_of_s(s, kp, Ti, Td, N):
    # C(s) = kp*(1 + 1/(Ti s) + (Td s)/(1 + Td s/N))
    return kp * (1.0 + 1.0/(Ti*s) + (Td*s)/(1.0 + Td*s/N))

def G_of_s(s, m, d, k):
    # G(s) = 1 / (m s^2 + d s + k)
    return 1.0 / (m*s**2 + d*s + k)

# ----------- time simulation -----------
T_end = 15.0
t = np.arange(0.0, T_end + Ts, Ts)
r_steps = np.ones(6)  # unit steps on all axes

eta = np.zeros((len(t), 6))   # output
nu  = np.zeros((len(t), 6))   # rate
u   = np.zeros((len(t), 6))   # control
e   = np.zeros((len(t), 6))   # error
I_int  = np.zeros(6)          # integral state
d_filt = np.zeros(6)          # filtered derivative of error

def update_pid(err, derr, kp, Ti, Td, N, I_state, d_state):
    # D filter: tau_d = Td/N; exact 1st-order discrete update
    tau_d = Td / N if N > 0 and Td > 0 else 0.0
    if tau_d > 0:
        alpha_d = Ts / (Ts + tau_d)
        d_state = d_state + alpha_d * (derr - d_state)
        d_term  = Td * d_state
    else:
        d_term  = 0.0
    I_state = I_state + (kp / Ti) * err * Ts
    u_cmd   = kp * (err + I_state / kp + d_term)
    return u_cmd, I_state, d_state

for k in range(1, len(t)):
    for i in range(6):
        r = r_steps[i]
        e_k  = r - eta[k-1, i]
        de_k = (e_k - e[k-1, i]) / Ts
        u_cmd, I_int[i], d_filt[i] = update_pid(
            e_k, de_k, Kp_all[i], Ti_all[i], Td_all[i], N, I_int[i], d_filt[i]
        )
        u[k, i] = u_cmd
        e[k, i] = e_k

        m_i = masses[i]; d_i = damps[i]; k_i = stiff[i]
        acc = (u_cmd - d_i*nu[k-1, i] - k_i*eta[k-1, i]) / m_i
        nu[k, i]  = nu[k-1, i] + Ts * acc
        eta[k, i] = eta[k-1, i] + Ts * nu[k, i]

# ----------- plots: step, bode, nyquist -----------
axis_names = ["Surge x", "Sway y", "Heave z", "Roll φ", "Pitch θ", "Yaw ψ"]
step_characteristics = [compute_step_characteristics(t, eta[:, i]) for i in range(6)]
color_map = plt.cm.get_cmap("tab10", len(axis_names))

# step responses (combined)
fig = plt.figure(figsize=(10, 6))
for i, name in enumerate(axis_names):
    plt.plot(t, eta[:, i], label=name, color=color_map(i))
plt.plot(t, np.ones_like(t), linestyle='--', color='black', label='Reference')
plt.xlabel("Time [s]")
plt.ylabel("Response")
plt.title("PID Step Responses - All Axes")
plt.grid(True)
plt.legend()
fig.tight_layout()
fig.savefig(os.path.join(outdir, "step_all_axes.png"), dpi=150, bbox_inches='tight')
plt.close(fig)

# individual step responses
for i, name in enumerate(axis_names):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(t, eta[:, i], label="Response", color=color_map(i))
    plt.plot(t, np.ones_like(t), linestyle='--', color='black', label='Reference')
    plt.xlabel("Time [s]")
    plt.ylabel("Response")
    plt.title(f"PID Step Response - {name}")
    plt.grid(True)
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"step_{i+1}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

# frequency response
w = np.logspace(-2, 2, 300)
jw = 1j*w
loop_margins = []
open_loop_responses = []
bode_magnitudes = []
bode_phases = []

for i in range(len(axis_names)):
    kp, Ti, Td = Kp_all[i], Ti_all[i], Td_all[i]
    m_i, d_i, k_i = masses[i], damps[i], stiff[i]

    L = C_of_s(jw, kp, Ti, Td, N) * G_of_s(jw, m_i, d_i, k_i)
    open_loop_responses.append(L)
    bode_magnitudes.append(20*np.log10(np.abs(L)))
    bode_phases.append(np.angle(L, deg=True))
    margins = compute_margins(w, L)
    loop_margins.append(margins)

# combined Bode plots
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ax_mag, ax_phase = axes
for i, name in enumerate(axis_names):
    color = color_map(i)
    ax_mag.semilogx(w, bode_magnitudes[i], label=name, color=color)
    ax_phase.semilogx(w, bode_phases[i], label=name, color=color)

ax_mag.set_ylabel("Magnitude [dB]")
ax_mag.set_title("PID Bode Magnitude - All Axes")
ax_mag.grid(True, which="both")

ax_phase.set_xlabel("Frequency [rad/s]")
ax_phase.set_ylabel("Phase [deg]")
ax_phase.set_title("PID Bode Phase - All Axes")
ax_phase.grid(True, which="both")
ax_phase.legend(loc="best")

fig.tight_layout()
fig.savefig(os.path.join(outdir, "bode_all_axes.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# combined Nyquist plot
fig = plt.figure(figsize=(8, 8))
for i, name in enumerate(axis_names):
    color = color_map(i)
    L = open_loop_responses[i]
    plt.plot(np.real(L), np.imag(L), label=name, color=color)
    margins = loop_margins[i]
    if margins["gain_crossover_freq"] is not None:
        L_gc = C_of_s(
            1j * margins["gain_crossover_freq"], Kp_all[i], Ti_all[i], Td_all[i], N
        ) * G_of_s(1j * margins["gain_crossover_freq"], masses[i], damps[i], stiff[i])
        plt.scatter(np.real(L_gc), np.imag(L_gc), color=color, marker="o", zorder=5)
    if margins["phase_crossover_freq"] is not None:
        L_pc = C_of_s(
            1j * margins["phase_crossover_freq"], Kp_all[i], Ti_all[i], Td_all[i], N
        ) * G_of_s(1j * margins["phase_crossover_freq"], masses[i], damps[i], stiff[i])
        plt.scatter(np.real(L_pc), np.imag(L_pc), color=color, marker="s", zorder=5)
plt.plot([-1], [0], marker='x', color='black')
plt.xlabel("Re{L(jω)}")
plt.ylabel("Im{L(jω)}")
plt.title("PID Nyquist Plot - All Axes")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.xlim(-20, 20)
plt.ylim(-20, 20)
fig.tight_layout()
fig.savefig(os.path.join(outdir, "nyquist_all_axes.png"), dpi=150, bbox_inches='tight')
plt.close(fig)

# per-axis Bode and Nyquist plots
for i, name in enumerate(axis_names):
    L = open_loop_responses[i]
    mag = bode_magnitudes[i]
    phs = bode_phases[i]
    margins = loop_margins[i]

    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    ax0, ax1, ax2 = axes

    ax0.semilogx(w, mag, color=color_map(i))
    ax0.set_title(f"Bode Magnitude - {name}")
    ax0.set_xlabel("Frequency [rad/s]")
    ax0.set_ylabel("Magnitude [dB]")
    ax0.grid(True, which="both")
    if margins["gain_crossover_freq"] is not None:
        ax0.scatter(margins["gain_crossover_freq"], 0.0, color="red", marker="o", zorder=5)
        pm = margins["phase_margin"]
        pm_text = f"{pm:.1f}°" if pm is not None else "N/A"
        ax0.annotate(
            f"ωgc={margins['gain_crossover_freq']:.2f}, PM={pm_text}",
            xy=(margins["gain_crossover_freq"], 0.0),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color="red",
        )

    ax1.semilogx(w, phs, color=color_map(i))
    ax1.set_title(f"Bode Phase - {name}")
    ax1.set_xlabel("Frequency [rad/s]")
    ax1.set_ylabel("Phase [deg]")
    ax1.grid(True, which="both")
    if margins["gain_crossover_freq"] is not None:
        ph_at_gc = np.angle(
            C_of_s(1j * margins["gain_crossover_freq"], Kp_all[i], Ti_all[i], Td_all[i], N)
            * G_of_s(1j * margins["gain_crossover_freq"], masses[i], damps[i], stiff[i]),
            deg=True,
        )
        ax1.scatter(margins["gain_crossover_freq"], ph_at_gc, color="red", marker="o", zorder=5)
        pm = margins["phase_margin"]
        pm_text = f"PM={pm:.1f}°" if pm is not None else "PM=N/A"
        ax1.annotate(
            pm_text,
            xy=(margins["gain_crossover_freq"], ph_at_gc),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            color="red",
        )
    if margins["phase_crossover_freq"] is not None:
        ax1.scatter(margins["phase_crossover_freq"], -180.0, color="blue", marker="s", zorder=5)
        gm_db = margins["gain_margin_db"]
        gm_text = f"GM={gm_db:.1f} dB" if gm_db is not None else "GM=N/A"
        ax1.annotate(
            gm_text,
            xy=(margins["phase_crossover_freq"], -180.0),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color="blue",
        )

    ax2.plot(np.real(L), np.imag(L), color=color_map(i))
    ax2.plot([-1], [0], marker='x', color='black')
    ax2.set_title(f"Nyquist Plot - {name}")
    ax2.set_xlabel("Re{L(jω)}")
    ax2.set_ylabel("Im{L(jω)}")
    ax2.grid(True)
    ax2.axis("equal")
    if margins["gain_crossover_freq"] is not None:
        s_gc = 1j * margins["gain_crossover_freq"]
        L_gc = C_of_s(s_gc, Kp_all[i], Ti_all[i], Td_all[i], N) * G_of_s(s_gc, masses[i], damps[i], stiff[i])
        ax2.scatter(np.real(L_gc), np.imag(L_gc), color="red", marker="o", zorder=5)
    if margins["phase_crossover_freq"] is not None:
        s_pc = 1j * margins["phase_crossover_freq"]
        L_pc = C_of_s(s_pc, Kp_all[i], Ti_all[i], Td_all[i], N) * G_of_s(s_pc, masses[i], damps[i], stiff[i])
        ax2.scatter(np.real(L_pc), np.imag(L_pc), color="blue", marker="s", zorder=5)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"frequency_{i+1}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

report_lines = []
report_lines.append("PID Step Response Characteristics\n")
for idx, name in enumerate(axis_names):
    metrics = step_characteristics[idx]
    margins = loop_margins[idx] if idx < len(loop_margins) else {}
    report_lines.append(f"Axis: {ascii_label(name)}")
    report_lines.append(
        "  Gains: "
        f"Kp={Kp_all[idx]:.3f}, Ki={Ki_all[idx]:.3f}, Kd={Kd_all[idx]:.3f}, "
        f"Ti={Ti_all[idx]:.3f}, Td={Td_all[idx]:.3f}"
    )
    report_lines.append(f"  Steady-state value: {metrics['steady_state']:.4f}")
    report_lines.append(f"  Overshoot: {metrics['overshoot']:.2f}%")
    settle = metrics['settling_time']
    report_lines.append(
        f"  Settling time (2% band): {settle:.3f} s" if settle is not None else "  Settling time (2% band): N/A"
    )
    rise = metrics['rise_time']
    report_lines.append(
        f"  Rise time (10-90%): {rise:.3f} s" if rise is not None else "  Rise time (10-90%): N/A"
    )
    zeta = metrics['damping_ratio']
    report_lines.append(
        f"  Damping ratio (est.): {zeta:.3f}" if zeta is not None else "  Damping ratio (est.): N/A"
    )
    if margins:
        gm_freq = margins["gain_crossover_freq"]
        pm = margins["phase_margin"]
        pc_freq = margins["phase_crossover_freq"]
        gm_db = margins["gain_margin_db"]
        report_lines.append(
            f"  Gain crossover: {gm_freq:.3f} rad/s" if gm_freq is not None else "  Gain crossover: N/A"
        )
        report_lines.append(
            f"  Phase margin: {pm:.2f} deg" if pm is not None else "  Phase margin: N/A"
        )
        report_lines.append(
            f"  Phase crossover: {pc_freq:.3f} rad/s" if pc_freq is not None else "  Phase crossover: N/A"
        )
        report_lines.append(
            f"  Gain margin: {gm_db:.2f} dB" if gm_db is not None else "  Gain margin: N/A"
        )
    report_lines.append("")

report_path = os.path.join(outdir, "pid_response_report.txt")
with open(report_path, "w", encoding="utf-8") as report_file:
    report_file.write("\n".join(report_lines))

print("Saved to:", outdir)
print("Report:", report_path)
