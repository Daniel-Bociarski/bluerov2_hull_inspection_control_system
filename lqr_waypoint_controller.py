#!/usr/bin/env python3


from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

def quaternion_to_euler(x: float, y: float, z: float, w: float):
    t0 = +2.0 * (w * x + y * z); t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x); t2 = +1.0 if t2 > +1.0 else t2; t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y); t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw

def shortest_angle(target: float, current: float) -> float:
    d = target - current
    while d > math.pi: d -= 2*math.pi
    while d < -math.pi: d += 2*math.pi
    return d

def aug_ss_cont(m: float, d: float, k: float):
    a = np.array([[0.0, 1.0, 0.0], [-k / m, -d / m, 0.0], [-1.0, 0.0, 0.0]])
    b = np.array([[0.0], [1.0 / m], [0.0]])
    return a, b

def c2d_euler(a: np.ndarray, b: np.ndarray, ts: float):
    ad = np.eye(a.shape[0]) + ts * a
    bd = ts * b
    return ad, bd

def dlqr(ad: np.ndarray, bd: np.ndarray, q: np.ndarray, r: np.ndarray, max_iter: int = 5000, eps: float = 1e-9):
    p = q.copy()
    for _ in range(max_iter):
        k = np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)
        pn = ad.T @ p @ (ad - bd @ k) + q
        if np.linalg.norm(pn - p, ord='fro') <= eps: p = pn; break
        p = pn
    return np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)

@dataclass
class AxisConfig:
    name: str
    k: np.ndarray
    max_output: float
    integral_limit: float

class LQRWaypointController(Node):
    def __init__(self):
        super().__init__('lqr_waypoint_controller')

        # faster sway + yaw both directions
        self.declare_parameter('control_hz', 30.0)
        self.declare_parameter('max_linear', 0.30)         # was 0.15
        self.declare_parameter('max_vertical', 0.12)
        self.declare_parameter('max_yaw_rate', 0.25)       # was 0.2
        self.declare_parameter('integral_limit', 0.6)
        self.declare_parameter('integral_decay', 0.93)
        self.declare_parameter('command_smoothing', 0.10)  # was 0.2
        self.declare_parameter('error_deadband_linear', 0.01)  # was 0.02
        self.declare_parameter('error_deadband_vertical', 0.02)
        self.declare_parameter('error_deadband_yaw', 0.015)

        self.control_hz = float(self.get_parameter('control_hz').value)
        self.dt = 1.0 / self.control_hz
        self.max_linear = float(self.get_parameter('max_linear').value)
        self.max_vertical = float(self.get_parameter('max_vertical').value)
        self.max_yaw_rate = float(self.get_parameter('max_yaw_rate').value)
        self.integral_limit = float(self.get_parameter('integral_limit').value)
        self.integral_decay = float(self.get_parameter('integral_decay').value)
        self.deadband_linear = float(self.get_parameter('error_deadband_linear').value)
        self.deadband_vertical = float(self.get_parameter('error_deadband_vertical').value)
        self.deadband_yaw = float(self.get_parameter('error_deadband_yaw').value)
        smoothing = float(self.get_parameter('command_smoothing').value)
        self.command_smoothing = max(0.0, min(1.0, smoothing))

        self.axes: Dict[str, AxisConfig] = self._compute_lqr_gains(self.dt)
        self.integral: Dict[str, float] = {axis: 0.0 for axis in self.axes}
        self.reference: Dict[str, float] = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}

        self.last_pose: Optional[PoseStamped] = None
        self.last_twist: Optional[Twist] = None
        self.prev_cmd = Twist()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/model/orca4/odometry', self._on_odometry, 10)
        self.create_subscription(PoseStamped, '/lqr_waypoint', self._on_waypoint, 10)
        self.timer = self.create_timer(self.dt, self._on_timer)

        self.get_logger().info(f'LQR waypoint controller running at {self.control_hz:.1f} Hz with dt={self.dt:.3f}s')

    def _compute_lqr_gains(self, ts: float) -> Dict[str, AxisConfig]:
        m_diag = np.array([5.5, 1.7, 3.57, 0.14, 0.11, 0.25], dtype=float)
        d_diag = np.array([4.03, 6.22, 5.18, 0.07, 0.07, 0.07], dtype=float)
        g = 9.81; m = 11.0
        k_phi_theta = -(-0.01) * m * g
        k_diag = np.array([0.0, 0.0, 0.0, k_phi_theta, k_phi_theta, 0.0], dtype=float)

        q_list = [
            np.diag([8.0, 0.5, 3.0]),
            np.diag([10.0, 0.6, 3.0]),
            np.diag([6.0, 0.4, 2.5]),
            np.diag([12.0, 0.8, 6.0]),
            np.diag([14.0, 0.8, 8.0]),
            np.diag([10.0, 0.7, 5.0]),
        ]
        r_list = [
            np.array([[1.0]]),   # more sway authority
            np.array([[1.0]]),
            np.array([[1.2]]),
            np.array([[0.45]]),
            np.array([[0.55]]),
            np.array([[9.0]]),
        ]

        gain_map: Dict[str, AxisConfig] = {}
        axis_lookup = {'x':0, 'y':1, 'z':2, 'roll':3, 'pitch':4, 'yaw':5}
        for axis, idx in axis_lookup.items():
            ad, bd = c2d_euler(*aug_ss_cont(m_diag[idx], d_diag[idx], k_diag[idx]), ts)
            k = dlqr(ad, bd, q_list[idx], r_list[idx])

            if axis in ('x','y'): max_out = self.max_linear
            elif axis == 'z':     max_out = self.max_vertical
            elif axis == 'yaw':   max_out = self.max_yaw_rate
            else:                 max_out = 0.0

            integral_limit = self.integral_limit * (0.5 if axis == 'yaw' else 0.7 if axis == 'z' else 1.0)
            gain_map[axis] = AxisConfig(axis, k.reshape(-1), max_out, integral_limit)

            if axis in {'x','y','z','yaw'}:
                self.get_logger().info(f'Axis {axis} gains: {gain_map[axis].k[0]:.3f}, {gain_map[axis].k[1]:.3f}, {gain_map[axis].k[2]:.3f}')
        return gain_map

    def _on_waypoint(self, msg: PoseStamped):
        self.reference['x'] = msg.pose.position.x
        self.reference['y'] = msg.pose.position.y
        self.reference['z'] = msg.pose.position.z
        _, _, yaw = quaternion_to_euler(msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)
        self.reference['yaw'] = yaw
        for axis in self.integral: self.integral[axis] = 0.0
        self.get_logger().info(f'New waypoint received: ({self.reference["x"]:.2f}, {self.reference["y"]:.2f}, {self.reference["z"]:.2f}, yaw={math.degrees(self.reference["yaw"]):.1f}°)')

    def _on_odometry(self, msg: Odometry):
        pose = PoseStamped(); pose.header = msg.header; pose.pose = msg.pose.pose
        self.last_pose = pose; self.last_twist = msg.twist.twist

    def _deadband(self, axis: str, value: float) -> float:
        thr = self.deadband_linear if axis in {'x','y'} else self.deadband_vertical if axis=='z' else self.deadband_yaw if axis=='yaw' else 0.0
        return 0.0 if abs(value) <= thr else value

    def _update_axis(self, axis: str, position_error: float, velocity: float) -> float:
        cfg = self.axes[axis]
        e = self._deadband(axis, position_error)
        if e == 0.0: self.integral[axis] *= self.integral_decay
        else:
            self.integral[axis] = max(-cfg.integral_limit, min(cfg.integral_limit, self.integral[axis] + (-e) * self.dt))
        u = float(-(cfg.k @ np.array([e, velocity, self.integral[axis]])))
        if cfg.max_output > 0.0: u = max(-cfg.max_output, min(cfg.max_output, u))
        return u

    def _on_timer(self):
        if self.last_pose is None or self.last_twist is None: return
        pose = self.last_pose.pose; twist = self.last_twist
        _, _, yaw = quaternion_to_euler(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)

        # position error in body frame
        ex_w = pose.position.x - self.reference['x']; ey_w = pose.position.y - self.reference['y']
        cy = math.cos(yaw); sy = math.sin(yaw)
        ex_b = cy*ex_w + sy*ey_w; ey_b = -sy*ex_w + cy*ey_w

        # IMPORTANT: yaw error sign -> shortest path with correct rotation sense
        # Use current-target, not target-current, to align with control sign.
        yaw_err = shortest_angle(yaw, self.reference['yaw'])

        errors = {'x': ex_b, 'y': ey_b, 'z': pose.position.z - self.reference['z'], 'yaw': yaw_err}
        vels   = {'x': twist.linear.x, 'y': twist.linear.y, 'z': twist.linear.z, 'yaw': twist.angular.z}

        raw = Twist()
        raw.linear.x  = self._update_axis('x',   errors['x'],  vels['x'])
        raw.linear.y  = self._update_axis('y',   errors['y'],  vels['y'])
        raw.linear.z  = self._update_axis('z',   errors['z'],  vels['z'])
        raw.angular.z = self._update_axis('yaw', errors['yaw'], vels['yaw'])

        if self.command_smoothing > 0.0:
            a = self.command_smoothing; cmd = Twist()
            cmd.linear.x  = self.prev_cmd.linear.x  + a*(raw.linear.x  - self.prev_cmd.linear.x)
            cmd.linear.y  = self.prev_cmd.linear.y  + a*(raw.linear.y  - self.prev_cmd.linear.y)
            cmd.linear.z  = self.prev_cmd.linear.z  + a*(raw.linear.z  - self.prev_cmd.linear.z)
            cmd.angular.z = self.prev_cmd.angular.z + a*(raw.angular.z - self.prev_cmd.angular.z)
        else:
            cmd = raw

        self.prev_cmd = Twist()
        self.prev_cmd.linear.x, self.prev_cmd.linear.y, self.prev_cmd.linear.z = cmd.linear.x, cmd.linear.y, cmd.linear.z
        self.prev_cmd.angular.z = cmd.angular.z
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init(); node = LQRWaypointController()
    try: rclpy.spin(node)
    except KeyboardInterrupt: node.get_logger().info('LQR waypoint controller interrupted by user.')
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
