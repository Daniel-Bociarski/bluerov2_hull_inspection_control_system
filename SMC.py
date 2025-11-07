#!/usr/bin/env python3
# SMC closed-loop simulation for BlueROV2 axes

import numpy as np
import matplotlib.pyplot as plt
import os



Ts = 0.01
T_end = 15.0
# Set reasonable control limits
u_max = np.array([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
w_ref = [None, None, None, None, 0.5, None]


g = 9.81
m_sys = 11.0
B = m_sys * g
zB = -0.01
k_phi_theta = -zB * B

M_diag  = np.array([5.5,  1.7,  3.57, 0.14, 0.11, 0.25], dtype=float)
D1_diag = np.array([4.03, 6.22, 5.18, 0.07, 0.07, 0.07], dtype=float)
K_diag  = np.array([0.0,  0.0,  0.0,  k_phi_theta, k_phi_theta, 0.0], dtype=float)

axis_names = ["Surge x", "Sway y", "Heave z", "Roll φ", "Pitch θ", "Yaw ψ"]
color_map = plt.cm.get_cmap("tab10", len(axis_names))


c1 = np.ones(6, dtype=float)
c2 = np.array([1.2, 1.8, 1.2, 2.0, 2.0, 1.8], dtype=float)   # lambda (from theory)
c3 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)   # gamma (from theory)
phi = np.array([0.01, 0.01, 0.01, 0.0087266, 0.0087266, 0.0087266], dtype=float)  # boundary layer (from theory)

# Switching gain (from disturbance analysis tables in theory)
k_s = np.array([4.244, 16.916, 9.968, 16.520, 21.093, 16.824], dtype=float)



def sat(x):
    return np.clip(x, -1.0, 1.0)

def ascii_label(label):
    return label.replace("φ","phi").replace("θ","theta").replace("ψ","psi")

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

def G_of_s(s, m, d, k):
    return 1.0 / (m*s**2 + d*s + k)

def compute_margins(w, L):
    mag = np.abs(L); mag_db = 20*np.log10(mag + 1e-12)
    phase = np.unwrap(np.angle(L)); phase_deg = np.degrees(phase)
    gain_crossover_freq = None; phase_margin = None
    phase_crossover_freq = None; gain_margin_db = None
    for i in range(len(w)-1):
        if (mag_db[i])*(mag_db[i+1]) <= 0.0:
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
            mag_at = 20*np.log10(np.abs(np.interp(wpc, w, L)) + 1e-12)
            gain_margin_db = -mag_at
            break
    return dict(gain_crossover_freq=gain_crossover_freq, phase_margin=phase_margin,
                phase_crossover_freq=phase_crossover_freq, gain_margin_db=gain_margin_db)



t = np.arange(0.0, T_end+Ts, Ts)
eta = np.zeros((len(t),6))
nu  = np.zeros((len(t),6))
u   = np.zeros((len(t),6))
z_int = np.zeros(6)
r_step = np.ones(6)
r_filt = np.zeros(6)



for k in range(1, len(t)):
    # Reference filter
    for i in range(6):
        if w_ref[i] is None:
            r_filt[i] = r_step[i]
        else:
            tau = 1.0 / w_ref[i]
            alpha = Ts/(Ts + tau)
            r_filt[i] = r_filt[i] + alpha*(r_step[i] - r_filt[i])

    # Control law for each axis
    for i in range(6):
        m = M_diag[i]; d = D1_diag[i]; Kk = K_diag[i]
        
        # Tracking error
        e = r_filt[i] - eta[k-1,i]
        e_dot = -nu[k-1,i]  # reference velocity is zero
        
        # Sliding surface
        s = e_dot + c2[i]*e + c3[i]*z_int[i]
        
        desired_acc = -c2[i]*nu[k-1,i] + c3[i]*e + k_s[i]*sat(s / max(phi[i], 1e-6))
        #             ^^^ NEGATIVE!    ^^^ POSITIVE!

        ui = m*desired_acc + d*nu[k-1,i] + Kk*eta[k-1,i]
        
        # Control saturation
        ui = float(np.clip(ui, -u_max[i], u_max[i]))
        u[k,i] = ui
        
        # Anti-windup: only integrate when control is not saturated
        if abs(ui) < 0.95 * u_max[i]:  # Leave 5% margin
            z_int[i] += e * Ts
            # Add explicit integral limits for better stability
            z_int[i] = np.clip(z_int[i], -5.0, 5.0) if i < 3 else np.clip(z_int[i], -0.5, 0.5)
        
        # Plant dynamics
        acc = (ui - d*nu[k-1,i] - Kk*eta[k-1,i]) / m
        nu[k,i]  = nu[k-1,i] + Ts*acc
        eta[k,i] = eta[k-1,i] + Ts*nu[k,i]



script_dir = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(script_dir, "smc_analysis")
os.makedirs(outdir, exist_ok=True)

w = np.logspace(-2, 2, 300)
jw = 1j*w
loop_margins = []
open_loop_responses = []
bode_magnitudes = []
bode_phases = []
equivalent_gains = []

for i, name in enumerate(axis_names):
    m = M_diag[i]; d = D1_diag[i]; Kk = K_diag[i]
    
    # Equivalent gains for linearized analysis
    Kp_eq = m * c2[i] * c3[i] + Kk
    Ki_eq = m * c3[i] * k_s[i] / phi[i]
    Kd_eq = m * (c2[i] * k_s[i] / phi[i] + c2[i]**2) + d
    
    Cjw = Kp_eq + Ki_eq/(jw + 1e-12) + Kd_eq*(jw)
    Ljw = Cjw * G_of_s(jw, m, d, Kk)
    open_loop_responses.append(Ljw)
    bode_magnitudes.append(20*np.log10(np.abs(Ljw) + 1e-12))
    bode_phases.append(np.angle(Ljw, deg=True))
    loop_margins.append(compute_margins(w, Ljw))
    equivalent_gains.append((Kp_eq, Ki_eq, Kd_eq))

fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ax_mag, ax_phase = axes
for i, name in enumerate(axis_names):
    color = color_map(i)
    ax_mag.semilogx(w, bode_magnitudes[i], label=name, color=color)
    ax_phase.semilogx(w, bode_phases[i], label=name, color=color)

ax_mag.set_ylabel("Magnitude [dB]")
ax_mag.set_title("SMC Bode Magnitude - All Axes")
ax_mag.grid(True, which="both")

ax_phase.set_xlabel("Frequency [rad/s]")
ax_phase.set_ylabel("Phase [deg]")
ax_phase.set_title("SMC Bode Phase - All Axes")
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
    Kp_eq, Ki_eq, Kd_eq = equivalent_gains[i]
    if margins["gain_crossover_freq"] is not None:
        s_gc = 1j * margins["gain_crossover_freq"]
        C_gc = Kp_eq + Ki_eq / s_gc + Kd_eq * s_gc
        L_gc = C_gc * G_of_s(s_gc, M_diag[i], D1_diag[i], K_diag[i])
        plt.scatter(np.real(L_gc), np.imag(L_gc), color=color, marker="o", zorder=5)
    if margins["phase_crossover_freq"] is not None:
        s_pc = 1j * margins["phase_crossover_freq"]
        C_pc = Kp_eq + Ki_eq / s_pc + Kd_eq * s_pc
        L_pc = C_pc * G_of_s(s_pc, M_diag[i], D1_diag[i], K_diag[i])
        plt.scatter(np.real(L_pc), np.imag(L_pc), color=color, marker="s", zorder=5)
plt.plot([-1], [0], marker='x', color='black')
plt.xlabel("Re{L(jω)}")
plt.ylabel("Im{L(jω)}")
plt.title("SMC Nyquist Plot - All Axes")
plt.grid(True)
plt.legend()
plt.axis('equal')
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
    Kp_eq, Ki_eq, Kd_eq = equivalent_gains[i]

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
        ph_at_gc = np.angle(
            (Kp_eq + Ki_eq / s_gc + Kd_eq * s_gc) * G_of_s(s_gc, M_diag[i], D1_diag[i], K_diag[i]),
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
    ax2.axis('equal')
    if margins["gain_crossover_freq"] is not None:
        s_gc = 1j * margins["gain_crossover_freq"]
        L_gc = (Kp_eq + Ki_eq / s_gc + Kd_eq * s_gc) * G_of_s(s_gc, M_diag[i], D1_diag[i], K_diag[i])
        ax2.scatter(np.real(L_gc), np.imag(L_gc), color="red", marker="o", zorder=5)
    if margins["phase_crossover_freq"] is not None:
        s_pc = 1j * margins["phase_crossover_freq"]
        L_pc = (Kp_eq + Ki_eq / s_pc + Kd_eq * s_pc) * G_of_s(s_pc, M_diag[i], D1_diag[i], K_diag[i])
        ax2.scatter(np.real(L_pc), np.imag(L_pc), color="blue", marker="s", zorder=5)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"frequency_{i+1}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Step response plots
# ============================================================

fig = plt.figure(figsize=(10, 6))
for i, name in enumerate(axis_names):
    plt.plot(t, eta[:, i], label=name, linewidth=2, color=color_map(i))
plt.plot(t, np.ones_like(t), "--", color="black", linewidth=2, alpha=0.7, label="Reference")
plt.xlabel("Time [s]")
plt.ylabel("Response")
plt.title("SMC Step Responses - All Axes (Corrected)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(outdir, "step_all_axes.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

for i, name in enumerate(axis_names):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(t, eta[:, i], label="Response", color=color_map(i))
    plt.plot(t, np.ones_like(t), "--", color="black", alpha=0.7, label="Reference")
    plt.xlabel("Time [s]")
    plt.ylabel("Response")
    plt.title(f"SMC Step Response - {name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"step_{i+1}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Control effort plots
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
for i, name in enumerate(axis_names):
    ax = axes[i//2, i%2]
    ax.plot(t, u[:, i], linewidth=1.5)
    ax.axhline(u_max[i], color='r', linestyle='--', alpha=0.5, label='limit')
    ax.axhline(-u_max[i], color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Control [N or Nm]")
    ax.set_title(f"Control Input - {name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(outdir, "control_inputs.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Report
# ============================================================

def report_line_metrics(i):
    metrics = compute_step_characteristics(t, eta[:, i])
    mrg = loop_margins[i]
    m = M_diag[i]; d = D1_diag[i]; Kk = K_diag[i]
    
    Kp_eq = m * c2[i] * c3[i] + Kk
    Ki_eq = m * c3[i] * k_s[i] / phi[i]
    Kd_eq = m * (c2[i] * k_s[i] / phi[i] + c2[i]**2) + d
    
    lines = []
    lines.append(f"Axis: {ascii_label(axis_names[i])}")
    lines.append(f"  SMC params: c2={c2[i]:.3f}, c3={c3[i]:.3f}, ks={k_s[i]:.3f}, phi={phi[i]:.3f}")
    lines.append(f"  Approx PID-equivalent: Kp={Kp_eq:.3f}, Ki={Ki_eq:.3f}, Kd={Kd_eq:.3f}")
    lines.append(f"  Control limit: ±{u_max[i]:.1f}")
    lines.append(f"  Steady-state value: {metrics['steady_state']:.4f}")
    lines.append(f"  Overshoot: {metrics['overshoot']:.2f}%")
    lines.append("  Settling time (2% band): " + (f"{metrics['settling_time']:.3f} s" if metrics['settling_time'] is not None else "N/A"))
    lines.append("  Rise time (10-90%): " + (f"{metrics['rise_time']:.3f} s" if metrics['rise_time'] is not None else "N/A"))
    lines.append("  Damping ratio (est.): " + (f"{metrics['damping_ratio']:.3f}" if metrics['damping_ratio'] is not None else "N/A"))
    lines.append("  Gain crossover: " + (f"{mrg['gain_crossover_freq']:.3f} rad/s" if mrg['gain_crossover_freq'] is not None else "N/A"))
    lines.append("  Phase margin: " + (f"{mrg['phase_margin']:.2f} deg" if mrg['phase_margin'] is not None else "N/A"))
    lines.append("  Phase crossover: " + (f"{mrg['phase_crossover_freq']:.3f} rad/s" if mrg['phase_crossover_freq'] is not None else "N/A"))
    lines.append("  Gain margin: " + (f"{mrg['gain_margin_db']:.2f} dB" if mrg['gain_margin_db'] is not None else "N/A"))
    lines.append("")
    return "\n".join(lines)

report = ["SMC Step Response Characteristics (FIXED VERSION)\n"]
report.append("Control Strategy: SMC with corrected signs and theory-based parameters")
report.append("Fixes: (1) Control law signs corrected, (2) Parameters from theory document")
report.append("Features: Proper model inversion, anti-windup, control saturation\n")
for i in range(6):
    report.append(report_line_metrics(i))

outdir_txt = os.path.join(outdir, "smc_response_report.txt")
with open(outdir_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"All figures and report saved in: {outdir}")
print(f"Report file: {outdir_txt}")
print("=" * 60)
