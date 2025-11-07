#!/usr/bin/env python3
# LQR closed-loop simulation for BlueROV2 axes (augmented with integral of position error)

import numpy as np
import matplotlib.pyplot as plt
import os

Ts = 0.01
T_end = 15.0
u_max = np.array([np.inf]*6)
w_ref = [None, None, None, None, 0.5, None]

Q_list = [
    np.diag([8.0, 0.5, 3.0]),
    np.diag([10.0, 0.6, 3.0]),
    np.diag([6.0, 0.4, 2.5]),
    np.diag([12.0, 0.8, 6.0]),
    np.diag([14.0, 0.8, 8.0]),
    np.diag([10.0, 0.7, 5.0]),
]
R_list = [np.array([[0.6]]), np.array([[0.7]]), np.array([[0.6]]),
          np.array([[0.15]]), np.array([[0.12]]), np.array([[0.18]])]

g = 9.81
m = 11.0
B = m * g
zB = -0.01
k_phi_theta = -zB * B

M_diag  = np.array([5.5,  1.7,  3.57, 0.14, 0.11, 0.25], dtype=float)
D1_diag = np.array([4.03, 6.22, 5.18, 0.07, 0.07, 0.07], dtype=float)
K_diag  = np.array([0.0,  0.0,  0.0,  k_phi_theta, k_phi_theta, 0.0], dtype=float)

axis_names = ["Surge x", "Sway y", "Heave z", "Roll φ", "Pitch θ", "Yaw ψ"]
color_map = plt.cm.get_cmap("tab10", len(axis_names))

def dlqr(Ad, Bd, Q, R, max_iter=5000, eps=1e-9):
    P = Q.copy()
    for _ in range(max_iter):
        BtPB = Bd.T @ P @ Bd
        K = np.linalg.solve(R + BtPB, Bd.T @ P @ Ad)
        Pn = Ad.T @ P @ (Ad - Bd @ K) + Q
        if np.linalg.norm(Pn - P, ord='fro') <= eps:
            P = Pn
            break
        P = Pn
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P

def aug_ss_cont(m, d, k):
    A = np.array([[0.0,        1.0, 0.0],
                  [-k/m,   -d/m, 0.0],
                  [-1.0,      0.0, 0.0]], dtype=float)
    B = np.array([[0.0],
                  [1.0/m],
                  [0.0]], dtype=float)
    C = np.array([[1.0, 0.0, 0.0]], dtype=float)
    return A, B, C

def c2d_euler(A, B, Ts):
    Ad = np.eye(A.shape[0]) + Ts*A
    Bd = Ts*B
    return Ad, Bd

K_list = []
Acl_list = []
for i in range(6):
    A, B, C = aug_ss_cont(M_diag[i], D1_diag[i], K_diag[i])
    Ad, Bd = c2d_euler(A, B, Ts)
    Kd, _ = dlqr(Ad, Bd, Q_list[i], R_list[i])
    K_list.append(Kd)
    Acl_list.append(Ad - Bd @ Kd)

t = np.arange(0.0, T_end+Ts, Ts)
eta = np.zeros((len(t),6))
nu  = np.zeros((len(t),6))
u   = np.zeros((len(t),6))
x = np.zeros((6,3))
r_step = np.ones(6)
r_filt = np.zeros(6)

for k in range(1, len(t)):
    for i in range(6):
        if w_ref[i] is None:
            r_filt[i] = r_step[i]
        else:
            tau = 1.0 / w_ref[i]
            alpha = Ts/(Ts + tau)
            r_filt[i] = r_filt[i] + alpha*(r_step[i] - r_filt[i])
    for i in range(6):
        e = r_filt[i] - x[i,0]
        x[i,2] += e * Ts
        ui = - (K_list[i] @ x[i]).item()
        if u_max[i] < np.inf:
            ui = float(np.clip(ui, -u_max[i], u_max[i]))
        u[k,i] = ui
        acc = (ui - D1_diag[i]*x[i,1] - K_diag[i]*x[i,0]) / M_diag[i]
        x[i,1] = x[i,1] + Ts*acc
        x[i,0] = x[i,0] + Ts*x[i,1]
        eta[k,i] = x[i,0]
        nu[k,i]  = x[i,1]

def linear_interpolate(x0, x1, y0, y1, target):
    if y1 == y0: return x0
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)

def compute_step_characteristics(t, y):
    yss = y[-1]; abs_final = max(abs(yss), 1e-9)
    Mp = (np.max(y)-yss)/abs_final*100.0
    tol = 0.02*abs_final
    err = np.abs(y - yss)
    ts = None
    for i in range(len(t)):
        if np.all(err[i:] <= tol):
            ts = t[i]; break
    t10=t90=None
    lo=0.1*yss; hi=0.9*yss
    for i in range(1,len(t)):
        if t10 is None and (y[i-1]-lo)*(y[i]-lo)<=0: t10=linear_interpolate(t[i-1],t[i],y[i-1],y[i],lo)
        if t90 is None and (y[i-1]-hi)*(y[i]-hi)<=0: t90=linear_interpolate(t[i-1],t[i],y[i-1],y[i],hi)
        if t10 is not None and t90 is not None: break
    tr = (t90-t10) if (t10 is not None and t90 is not None) else None
    zeta=None
    if Mp>0:
        ln_os=np.log(Mp/100.0)
        zeta = -ln_os/np.sqrt(np.pi**2+ln_os**2)
    return dict(steady_state=yss, overshoot=Mp, settling_time=ts, rise_time=tr, damping_ratio=zeta)

def ascii_label(label):
    return label.replace("φ","phi").replace("θ","theta").replace("ψ","psi")

def G_of_s(s, m, d, k):
    return 1.0 / (m*s**2 + d*s + k)

def compute_margins(w, L):
    mag = np.abs(L); mag_db = 20*np.log10(mag)
    phase = np.unwrap(np.angle(L)); phase_deg = np.degrees(phase)
    gain_crossover_freq = None; phase_margin = None
    phase_crossover_freq = None; gain_margin_db = None
    for i in range(len(w)-1):
        if (mag_db[i]-0.0)*(mag_db[i+1]-0.0) <= 0.0:
            wgc = linear_interpolate(w[i], w[i+1], mag_db[i], mag_db[i+1], 0.0)
            gain_crossover_freq = wgc
            ph_at = np.degrees(np.interp(wgc, w, phase))
            while ph_at <= -180.0: ph_at += 360.0
            while ph_at >   180.0: ph_at -= 360.0
            phase_margin = 180.0 + ph_at
            break
    for i in range(len(w)-1):
        if (phase_deg[i]+180.0)*(phase_deg[i+1]+180.0) <= 0.0:
            wpc = linear_interpolate(w[i], w[i+1], phase_deg[i], phase_deg[i+1], -180.0)
            phase_crossover_freq = wpc
            mag_at = 20*np.log10(np.abs(np.interp(wpc, w, L)))
            gain_margin_db = -mag_at
            break
    return dict(
        gain_crossover_freq=gain_crossover_freq,
        phase_margin=phase_margin,
        phase_crossover_freq=phase_crossover_freq,
        gain_margin_db=gain_margin_db,
    )

script_dir = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(script_dir, "lqr_analysis")
os.makedirs(outdir, exist_ok=True)

fig = plt.figure(figsize=(10, 6))
for i, name in enumerate(axis_names):
    plt.plot(t, eta[:, i], label=name, color=color_map(i))
plt.plot(t, np.ones_like(t), "--", color="black", label="Reference")
plt.xlabel("Time [s]")
plt.ylabel("Response")
plt.title("LQR Step Responses - All Axes")
plt.grid(True)
plt.legend()
fig.tight_layout()
fig.savefig(os.path.join(outdir, "step_all_axes.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

for i, name in enumerate(axis_names):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(t, eta[:, i], label="Response", color=color_map(i))
    plt.plot(t, np.ones_like(t), "--", color="black", label="Reference")
    plt.xlabel("Time [s]")
    plt.ylabel("Response")
    plt.title(f"LQR Step Response - {name}")
    plt.grid(True)
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"step_{i+1}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# Frequency-domain analysis: L(jw) = (k1 + ki/s + k2*s) * G(jw)
w = np.logspace(-2, 2, 300)
jw = 1j*w
loop_margins = []
open_loop_responses = []
bode_magnitudes = []
bode_phases = []
controller_params = []
for i in range(len(axis_names)):
    k1 = float(K_list[i][0,0])
    k2 = float(K_list[i][0,1])
    ki = float(K_list[i][0,2])
    Cjw = k1 + ki/(jw) + k2*(jw)
    Ljw = Cjw * G_of_s(jw, M_diag[i], D1_diag[i], K_diag[i])
    open_loop_responses.append(Ljw)
    bode_magnitudes.append(20*np.log10(np.abs(Ljw)))
    bode_phases.append(np.angle(Ljw, deg=True))
    loop_margins.append(compute_margins(w, Ljw))
    controller_params.append((k1, k2, ki))

fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ax_mag, ax_phase = axes
for i, name in enumerate(axis_names):
    color = color_map(i)
    ax_mag.semilogx(w, bode_magnitudes[i], label=name, color=color)
    ax_phase.semilogx(w, bode_phases[i], label=name, color=color)

ax_mag.set_ylabel("Magnitude [dB]")
ax_mag.set_title("LQR Bode Magnitude - All Axes")
ax_mag.grid(True, which="both")

ax_phase.set_xlabel("Frequency [rad/s]")
ax_phase.set_ylabel("Phase [deg]")
ax_phase.set_title("LQR Bode Phase - All Axes")
ax_phase.grid(True, which="both")
ax_phase.legend(loc="best")

fig.tight_layout()
fig.savefig(os.path.join(outdir, "bode_all_axes.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(8, 8))
for i, name in enumerate(axis_names):
    color = color_map(i)
    L = open_loop_responses[i]
    plt.plot(np.real(L), np.imag(L), label=name, color=color)
    margins = loop_margins[i]
    k1, k2, ki = controller_params[i]
    if margins["gain_crossover_freq"] is not None:
        s_gc = 1j * margins["gain_crossover_freq"]
        C_gc = k1 + ki / s_gc + k2 * s_gc
        L_gc = C_gc * G_of_s(s_gc, M_diag[i], D1_diag[i], K_diag[i])
        plt.scatter(np.real(L_gc), np.imag(L_gc), color=color, marker="o", zorder=5)
    if margins["phase_crossover_freq"] is not None:
        s_pc = 1j * margins["phase_crossover_freq"]
        C_pc = k1 + ki / s_pc + k2 * s_pc
        L_pc = C_pc * G_of_s(s_pc, M_diag[i], D1_diag[i], K_diag[i])
        plt.scatter(np.real(L_pc), np.imag(L_pc), color=color, marker="s", zorder=5)
plt.plot([-1], [0], marker="x", color="black")
plt.xlabel("Re{L(jω)}")
plt.ylabel("Im{L(jω)}")
plt.title("LQR Nyquist Plot - All Axes")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.xlim(-20, 20)
plt.ylim(-20, 20)
fig.tight_layout()
fig.savefig(os.path.join(outdir, "nyquist_all_axes.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

for i, name in enumerate(axis_names):
    L = open_loop_responses[i]
    mag = bode_magnitudes[i]
    phs = bode_phases[i]
    margins = loop_margins[i]
    k1, k2, ki = controller_params[i]

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
        s_gc = 1j * margins["gain_crossover_freq"]
        ph_at_gc = np.angle((k1 + ki / s_gc + k2 * s_gc) * G_of_s(s_gc, M_diag[i], D1_diag[i], K_diag[i]), deg=True)
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
    ax2.plot([-1], [0], marker="x", color="black")
    ax2.set_title(f"Nyquist Plot - {name}")
    ax2.set_xlabel("Re{L(jω)}")
    ax2.set_ylabel("Im{L(jω)}")
    ax2.grid(True)
    ax2.axis("equal")
    if margins["gain_crossover_freq"] is not None:
        s_gc = 1j * margins["gain_crossover_freq"]
        L_gc = (k1 + ki / s_gc + k2 * s_gc) * G_of_s(s_gc, M_diag[i], D1_diag[i], K_diag[i])
        ax2.scatter(np.real(L_gc), np.imag(L_gc), color="red", marker="o", zorder=5)
    if margins["phase_crossover_freq"] is not None:
        s_pc = 1j * margins["phase_crossover_freq"]
        L_pc = (k1 + ki / s_pc + k2 * s_pc) * G_of_s(s_pc, M_diag[i], D1_diag[i], K_diag[i])
        ax2.scatter(np.real(L_pc), np.imag(L_pc), color="blue", marker="s", zorder=5)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"frequency_{i+1}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def report_line_metrics(i):
    metrics = compute_step_characteristics(t, eta[:, i])
    ev = np.linalg.eigvals(Acl_list[i]); ev_mag = np.abs(ev)
    m = loop_margins[i]
    lines = []
    lines.append(f"Axis: {ascii_label(axis_names[i])}")
    lines.append(f"  LQR K: {np.array2string(K_list[i], precision=4, suppress_small=True)}")
    lines.append(f"  Steady-state value: {metrics['steady_state']:.4f}")
    lines.append(f"  Overshoot: {metrics['overshoot']:.2f}%")
    lines.append("  Settling time (2% band): " + (f"{metrics['settling_time']:.3f} s" if metrics['settling_time'] is not None else "N/A"))
    lines.append("  Rise time (10-90%): " + (f"{metrics['rise_time']:.3f} s" if metrics['rise_time'] is not None else "N/A"))
    lines.append("  Damping ratio (est.): " + (f"{metrics['damping_ratio']:.3f}" if metrics['damping_ratio'] is not None else "N/A"))
    lines.append(f"  Closed-loop Ad-BdK eigenvalues (magnitudes): {np.array2string(ev_mag, precision=4, suppress_small=True)}")
    lines.append("  Gain crossover: " + (f"{m['gain_crossover_freq']:.3f} rad/s" if m['gain_crossover_freq'] is not None else "N/A"))
    lines.append("  Phase margin: " + (f"{m['phase_margin']:.2f} deg" if m['phase_margin'] is not None else "N/A"))
    lines.append("  Phase crossover: " + (f"{m['phase_crossover_freq']:.3f} rad/s" if m['phase_crossover_freq'] is not None else "N/A"))
    lines.append("  Gain margin: " + (f"{m['gain_margin_db']:.2f} dB" if m['gain_margin_db'] is not None else "N/A"))
    lines.append("")
    return "\n".join(lines)

report = ["LQR Step Response Characteristics\n"]
for i in range(6):
    report.append(report_line_metrics(i))

outdir_txt = os.path.join(outdir, "lqr_response_report.txt")
with open(outdir_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("Saved to:", outdir)
print("Report:", outdir_txt)
