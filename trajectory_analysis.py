#!/usr/bin/env python3
"""Analyze ROV trajectory data stored in rov_pose_log.csv."""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


AxisData = Dict[str, np.ndarray]


def load_samples(csv_path: Path) -> Tuple[np.ndarray, AxisData]:
    stamps = []
    data: Dict[str, list] = {
        'x': [], 'y': [], 'z': [],
        'roll': [], 'pitch': [], 'yaw': []
    }

    with csv_path.open('r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            stamps.append(float(row['stamp_sec']) + float(row['stamp_nanosec']) * 1e-9)
            for key in data:
                data[key].append(float(row[key]))

    if not stamps:
        raise ValueError('No pose samples found in CSV')

    t = np.array(stamps)
    series = {key: np.array(vals) for key, vals in data.items()}
    return t, series


def compute_path_metrics(t: np.ndarray, axes: AxisData) -> Dict[str, float]:
    xs, ys, zs = axes['x'], axes['y'], axes['z']
    start = np.array([xs[0], ys[0], zs[0]])
    end = np.array([xs[-1], ys[-1], zs[-1]])

    displacements = np.diff(np.column_stack((xs, ys, zs)), axis=0)
    segment_lengths = np.linalg.norm(displacements, axis=1)
    path_length = float(segment_lengths.sum())
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    return {
        'start_x': start[0], 'start_y': start[1], 'start_z': start[2],
        'end_x': end[0], 'end_y': end[1], 'end_z': end[2],
        'displacement': float(np.linalg.norm(end - start)),
        'path_length': path_length,
        'duration': duration,
        'avg_speed': path_length / duration if duration > 0 else 0.0,
        'min_depth': float(np.max(zs)),  # z is negative down
        'max_depth': float(np.min(zs)),
    }


def steady_state_stats(values: np.ndarray, tail_fraction: float = 0.2) -> Tuple[float, float, float]:
    n_tail = max(int(len(values) * tail_fraction), 1)
    tail = values[-n_tail:]
    mean = float(np.mean(tail))
    std = float(np.std(tail))
    amplitude = float(0.5 * (np.max(tail) - np.min(tail)))
    return mean, std, amplitude


def settling_time(t: np.ndarray, values: np.ndarray, steady: float, tol: float = 0.02) -> float:
    band = max(tol, abs(steady) * tol)
    for idx in range(len(values)):
        if np.all(np.abs(values[idx:] - steady) <= band):
            return float(t[idx] - t[0])
    return float('nan')


def oscillation_frequency(t: np.ndarray, values: np.ndarray, steady: float) -> float:
    tail_len = max(int(len(values) * 0.3), 3)
    tail = values[-tail_len:] - steady
    if np.allclose(tail, 0.0):
        return float('nan')

    times = t[-tail_len:]
    signs = np.sign(tail)
    zero_crossings = []
    for i in range(1, len(signs)):
        if signs[i] == 0:
            zero_crossings.append(times[i])
        elif signs[i - 1] == 0:
            zero_crossings.append(times[i - 1])
        elif signs[i] != signs[i - 1]:
            t0, t1 = times[i - 1], times[i]
            y0, y1 = tail[i - 1], tail[i]
            tau = t0 + (0 - y0) * (t1 - t0) / (y1 - y0)
            zero_crossings.append(tau)

    if len(zero_crossings) < 3:
        return float('nan')

    zero_crossings = np.array(zero_crossings)
    periods = 2 * np.diff(zero_crossings)
    mean_period = np.mean(periods)
    return 2 * math.pi / mean_period if mean_period > 0 else float('nan')


def analyze_axes(t: np.ndarray, axes: AxisData) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.0

    for axis, values in axes.items():
        steady, std, amplitude = steady_state_stats(values)
        metrics[axis] = {
            'steady': steady,
            'std': std,
            'amplitude': amplitude,
            'settling_time': settling_time(t, values, steady) if dt > 0 else float('nan'),
            'osc_freq': oscillation_frequency(t, values, steady),
        }
    return metrics


def print_full_report(path_metrics: Dict[str, float], axis_metrics: Dict[str, Dict[str, float]]):
    print('=== Trajectory Overview ===')
    print(f"Start position:  ({path_metrics['start_x']: .3f}, {path_metrics['start_y']: .3f}, {path_metrics['start_z']: .3f}) m")
    print(f"End position:    ({path_metrics['end_x']: .3f}, {path_metrics['end_y']: .3f}, {path_metrics['end_z']: .3f}) m")
    print(f"Displacement:     {path_metrics['displacement']: .3f} m")
    print(f"Path length:      {path_metrics['path_length']: .3f} m")
    print(f"Mission duration: {path_metrics['duration']: .2f} s")
    print(f"Average speed:    {path_metrics['avg_speed']: .3f} m/s")
    print(f"Shallowest depth: {path_metrics['min_depth']: .3f} m")
    print(f"Deepest depth:    {path_metrics['max_depth']: .3f} m")
    print()

    headers = ('Axis', 'Steady', 'StdDev', 'Peak Δ', 'Settle [s]', 'Osc ω [rad/s]')
    print('=== Steady-State & Oscillation Metrics ===')
    print(f"{headers[0]:<6} {headers[1]:>10} {headers[2]:>10} {headers[3]:>10} {headers[4]:>12} {headers[5]:>14}")

    for axis in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']:
        m = axis_metrics[axis]
        steady = m['steady']
        std = m['std']
        amp = m['amplitude']
        settle = m['settling_time']
        osc = m['osc_freq']

        settle_str = f"{settle: .2f}" if math.isfinite(settle) else '   n/a'
        osc_str = f"{osc: .2f}" if math.isfinite(osc) else '       n/a'

        print(f"{axis:<6} {steady:>10.4f} {std:>10.4f} {amp:>10.4f} {settle_str:>12} {osc_str:>14}")
    print()


def _set_equal_aspect(ax, x, y):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    dx = xmax - xmin
    dy = ymax - ymin
    d = max(dx, dy)
    cx = 0.5 * (xmax + xmin)
    cy = 0.5 * (ymax + ymin)
    ax.set_xlim(cx - 0.55 * d, cx + 0.55 * d)
    ax.set_ylim(cy - 0.55 * d, cy + 0.55 * d)
    ax.set_aspect('equal', adjustable='box')


def plot_trajectory(t: np.ndarray, axes: AxisData, show_orientation: bool):
    xs, ys, zs = axes['x'], axes['y'], axes['z']
    rolls, pitches, yaws = axes['roll'], axes['pitch'], axes['yaw']

    # 3D path
    fig3d = plt.figure(figsize=(10, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.plot(xs, ys, zs, label='ROV Path')
    ax3d.scatter(xs[0], ys[0], zs[0], marker='o', label='Start')
    ax3d.scatter(xs[-1], ys[-1], zs[-1], marker='x', label='End')
    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Z [m]')
    ax3d.set_title('ROV Trajectory (3D)')
    ax3d.legend()
    ax3d.grid(True)

    # 2D plane subplots
    fig2d, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=False, sharey=False)
    # XY
    axs[0].plot(xs, ys)
    axs[0].scatter([xs[0]], [ys[0]], marker='o', label='Start')
    axs[0].scatter([xs[-1]], [ys[-1]], marker='x', label='End')
    axs[0].set_xlabel('X [m]')
    axs[0].set_ylabel('Y [m]')
    axs[0].set_title('XY plane')
    axs[0].grid(True)
    _set_equal_aspect(axs[0], xs, ys)
    axs[0].legend(loc='best')

    # XZ
    axs[1].plot(xs, zs)
    axs[1].scatter([xs[0]], [zs[0]], marker='o', label='Start')
    axs[1].scatter([xs[-1]], [zs[-1]], marker='x', label='End')
    axs[1].set_xlabel('X [m]')
    axs[1].set_ylabel('Z [m]')
    axs[1].set_title('XZ plane')
    axs[1].grid(True)
    _set_equal_aspect(axs[1], xs, zs)
    axs[1].legend(loc='best')

    # YZ
    axs[2].plot(ys, zs)
    axs[2].scatter([ys[0]], [zs[0]], marker='o', label='Start')
    axs[2].scatter([ys[-1]], [zs[-1]], marker='x', label='End')
    axs[2].set_xlabel('Y [m]')
    axs[2].set_ylabel('Z [m]')
    axs[2].set_title('YZ plane')
    axs[2].grid(True)
    _set_equal_aspect(axs[2], ys, zs)
    axs[2].legend(loc='best')

    fig2d.tight_layout()

    if show_orientation:
        sample_idx = np.arange(len(rolls))
        fig_orient, axes_plot = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes_plot[0].plot(sample_idx, rolls, label='Roll')
        axes_plot[0].set_ylabel('Roll [rad]')
        axes_plot[0].grid(True)
        axes_plot[1].plot(sample_idx, pitches, label='Pitch')
        axes_plot[1].set_ylabel('Pitch [rad]')
        axes_plot[1].grid(True)
        axes_plot[2].plot(sample_idx, yaws, label='Yaw')
        axes_plot[2].set_ylabel('Yaw [rad]')
        axes_plot[2].set_xlabel('Sample')
        axes_plot[2].grid(True)
        fig_orient.suptitle('Orientation vs Sample')
        fig_orient.tight_layout()

    plt.show()


def main():
    default_csv = Path.home() / 'Documents' / 'Thesis' / 'Data' / 'test.csv'

    parser = argparse.ArgumentParser(description='Analyze ROV trajectory CSV log')
    parser.add_argument('csv_path', nargs='?', default=default_csv, type=Path,
                        help='Path to pose CSV (defaults to ~/Documents/Thesis/Data/rov_pose_log.csv)')
    parser.add_argument('--plot', action='store_true', help='Show trajectory and orientation plots')
    args = parser.parse_args()

    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV file not found: {csv_path}')

    t, axes = load_samples(csv_path)
    path_metrics = compute_path_metrics(t, axes)
    axis_metrics = analyze_axes(t, axes)
    print_full_report(path_metrics, axis_metrics)

    if args.plot:
        plot_trajectory(t, axes, show_orientation=True)


if __name__ == '__main__':
    main()

