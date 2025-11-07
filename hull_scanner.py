#!/usr/bin/env python3


import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from orca_msgs.action import TargetMode
from rclpy.action import ActionClient
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Header


class SendGoalResult(Enum):
    SUCCESS = 0
    FAILURE = 1
    CANCELED = 2


@dataclass(frozen=True)
class HullWaypoint:
    point: Tuple[float, float, float]
    yaw: float  # yaw to hold while traveling to the NEXT waypoint


def make_pose(x: float, y: float, z: float) -> PoseStamped:
    return PoseStamped(
        header=Header(frame_id="map"),
        pose=Pose(position=Point(x=x, y=y, z=z)),
    )


class YawKalmanFilter:
    def __init__(self):
        self.x = 0.0
        self.bias = 0.0
        self.P = [[1.0, 0.0], [0.0, 1.0]]
        self.last_time: Optional[float] = None
        self.process_noise = 1e-3
        self.bias_noise = 1e-5
        self.measurement_noise = 1e-2
        self.initialized = False

    def reset(self):
        self.__init__()

    def update(self, yaw_measurement: float, gyro_z: float, stamp_sec: float):
        if self.last_time is None:
            self.x = yaw_measurement
            self.last_time = stamp_sec
            self.initialized = True
            return

        dt = max(stamp_sec - self.last_time, 1e-3)
        self.last_time = stamp_sec

        rate = gyro_z - self.bias
        self.x += dt * rate
        self.P[0][0] += dt * (dt * self.P[1][1] - self.P[1][0] - self.P[0][1] + self.process_noise)
        self.P[0][1] -= dt * self.P[1][1]
        self.P[1][0] -= dt * self.P[1][1]
        self.P[1][1] += self.bias_noise * dt

        innovation = yaw_measurement - self.x
        S = self.P[0][0] + self.measurement_noise
        K0 = self.P[0][0] / S
        K1 = self.P[1][0] / S

        self.x += K0 * innovation
        self.bias += K1 * innovation

        p00, p01 = self.P[0]
        p10, p11 = self.P[1]
        self.P[0][0] = p00 - K0 * p00
        self.P[0][1] = p01 - K0 * p01
        self.P[1][0] = p10 - K1 * p00
        self.P[1][1] = p11 - K1 * p01

    def yaw(self) -> Optional[float]:
        return self.x if self.initialized else None


def send_goal(node, action_client, goal_msg) -> SendGoalResult:
    goal_handle = None
    try:
        action_client.wait_for_server()
        goal_future = action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(node, goal_future)
        goal_handle = goal_future.result()

        if goal_handle is None:
            raise RuntimeError(f"Exception while sending goal: {goal_future.exception()}")
        if not goal_handle.accepted:
            node.get_logger().error("Goal rejected")
            return SendGoalResult.FAILURE

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)
        result = result_future.result()
        if result is None:
            raise RuntimeError(f"Exception while getting result: {result_future.exception()}")

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            node.get_logger().info("Goal completed")
            return SendGoalResult.SUCCESS
        if result.status == GoalStatus.STATUS_CANCELED:
            node.get_logger().warning("Goal canceled by action server")
            return SendGoalResult.CANCELED
        if result.status == GoalStatus.STATUS_ABORTED:
            node.get_logger().error("Goal aborted by action server")
            return SendGoalResult.FAILURE

        node.get_logger().error(f"Unexpected goal status: {result.status}")
        return SendGoalResult.FAILURE

    except KeyboardInterrupt:
        if goal_handle is not None and goal_handle.status in (
            GoalStatus.STATUS_ACCEPTED,
            GoalStatus.STATUS_EXECUTING,
        ):
            node.get_logger().info("Canceling goal...")
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, cancel_future)
        raise


class MissionState(Enum):
    INIT = auto()
    SET_AUV_MODE = auto()
    MOVE_TO_ORIGIN = auto()
    CHECK_ORIGIN = auto()
    MOVE_TO_WAYPOINT = auto()
    CHECK_WAYPOINT = auto()
    SET_ROV_MODE = auto()
    COMPLETE = auto()
    CANCELED = auto()
    ERROR = auto()


class HullScannerMission:
    ORIGIN_DEPTH = -0.3
    POSITION_TOLERANCE = 0.3
    ORIGIN_HOVER_TIME = 3.0
    WAYPOINT_HOVER_TIME = 1.0
    YAW_TOLERANCE = 0.1

    # second pass depth and center
    SECOND_DEPTH = -1.3
    CENTER_X = 6.75
    CENTER_Y = 8.5
    NUDGE_TOWARD_CENTER = 0.1  # 0.1 meters closer to center

    def __init__(self):
        self.node = rclpy.create_node("hull_scanner")
        self.logger = self.node.get_logger()
        self.set_target_mode = ActionClient(self.node, TargetMode, "/set_target_mode")
        self.state = MissionState.INIT

        self.go_auv_goal = TargetMode.Goal()
        self.go_auv_goal.target_mode = TargetMode.Goal.ORCA_MODE_AUV
        self.go_rov_goal = TargetMode.Goal()
        self.go_rov_goal.target_mode = TargetMode.Goal.ORCA_MODE_ROV

        self.current_position: Optional[Tuple[float, float, float]] = None
        self.current_orientation: Optional[Quaternion] = None
        self.pose_available = False
        self._pose_missing_logged = False

        self.current_target: Optional[Tuple[float, float, float]] = None
        self.current_yaw_target: Optional[float] = None

        self.hull_points = self._build_hull_points()
        self.waypoint_index = 0

        self.hover_start_time: Optional[float] = None
        self.last_log_time: Optional[float] = None

        self.sonar_range: Optional[float] = None
        self._sonar_ready_logged = False
        self._collision_reported = False

        self.yaw_filter = YawKalmanFilter()
        self._waiting_for_lqr = False

        qos = QoSProfile(depth=10)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.history = QoSHistoryPolicy.KEEP_LAST

        self.node.create_subscription(Odometry, "/model/orca4/odometry", self._on_odometry, qos)
        self.node.create_subscription(LaserScan, "/front_sonar/scan", self._on_sonar, qos)
        self.node.create_subscription(Imu, "/model/orca4/imu", self._on_imu, qos)
        self.waypoint_pub = self.node.create_publisher(PoseStamped, "/lqr_waypoint", 10)

    # --- ROS callbacks ---

    def _on_odometry(self, msg: Odometry):
        pos = msg.pose.pose.position
        self.current_position = (pos.x, pos.y, pos.z)
        self.current_orientation = msg.pose.pose.orientation
        if not self.pose_available:
            self.logger.info("Received first odometry sample.")
            self.pose_available = True
        self._pose_missing_logged = False

    def _on_sonar(self, msg: LaserScan):
        valid = [r for r in msg.ranges if msg.range_min <= r <= msg.range_max]
        if not valid:
            self.sonar_range = None
            return
        self.sonar_range = min(valid)
        if not self._sonar_ready_logged:
            self.logger.info("Sonar data available.")
            self._sonar_ready_logged = True
        if self.sonar_range <= 0.2:
            if not self._collision_reported:
                self.logger.warning(f"Collision warning: object detected at {self.sonar_range:.2f} m")
                self._collision_reported = True
        elif self._collision_reported and self.sonar_range > 0.25:
            self._collision_reported = False

    def _on_imu(self, msg: Imu):
        yaw = self._quat_to_yaw(msg.orientation)
        gyro_z = msg.angular_velocity.z
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.yaw_filter.update(yaw, gyro_z, stamp_sec)

    # --- Helper utilities ---

    def _current_time(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _distance_to_point(self, point: Tuple[float, float, float]) -> Optional[float]:
        if self.current_position is None:
            return None
        dx = self.current_position[0] - point[0]
        dy = self.current_position[1] - point[1]
        dz = self.current_position[2] - point[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _current_yaw(self) -> Optional[float]:
        filt = self.yaw_filter.yaw()
        if filt is not None:
            return filt
        if self.current_orientation is None:
            return None
        return self._quat_to_yaw(self.current_orientation)

    @staticmethod
    def _angular_difference(target: float, current: float) -> float:
        diff = target - current
        while diff > math.pi:
            diff -= 2.0 * math.pi
        while diff < -math.pi:
            diff += 2.0 * math.pi
        return diff

    @staticmethod
    def _quat_to_yaw(q: Quaternion) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _make_pose_with_yaw(x: float, y: float, z: float, yaw: float) -> PoseStamped:
        pose = make_pose(x, y, z)
        half = 0.5 * yaw
        pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))
        return pose

    def _set_target(self, target: Tuple[float, float, float], yaw: float):
        pose = self._make_pose_with_yaw(*target, yaw)
        self.waypoint_pub.publish(pose)
        self.current_target = target
        self.current_yaw_target = yaw
        self.logger.info(
            f"Waypoint: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}), yaw={math.degrees(yaw):.1f}°"
        )

    # --- Waypoint builder ---

    @staticmethod
    def _nudge_toward_center(x: float, y: float, cx: float, cy: float, dist: float) -> Tuple[float, float]:
        dx = cx - x
        dy = cy - y
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return x, y
        ux, uy = dx / norm, dy / norm
        return x + dist * ux, y + dist * uy

    @staticmethod
    def _shift_toward_origin(x: float, y: float, shift: float) -> Tuple[float, float]:
        """Shift a point toward origin by 'shift' meters in both X and Y directions"""
        # Move toward 0 in x direction
        if x > 0:
            new_x = max(0, x - shift)
        elif x < 0:
            new_x = min(0, x + shift)
        else:
            new_x = x
        
        # Move toward 0 in y direction
        if y > 0:
            new_y = max(0, y - shift)
        elif y < 0:
            new_y = min(0, y + shift)
        else:
            new_y = y
        
        return new_x, new_y

    def _build_hull_points(self) -> List[HullWaypoint]:
        z1 = self.ORIGIN_DEPTH
        z2 = self.SECOND_DEPTH

        seq: List[HullWaypoint] = []

        # Helper to append a point with a "yaw-after" value
        def add_point(x: float, y: float, z: float, yaw_after: float):
            seq.append(HullWaypoint(point=(round(x, 3), round(y, 3), z), yaw=yaw_after))

        # Helper to add a line - if diagonal, shift waypoints toward origin
        def add_line(x0: float, y0: float, x1: float, y1: float, z: float, travel_yaw: float, end_yaw: float):
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            n_steps = max(1, int(round(length)))
            
            # Check if this is a diagonal (both x and y change significantly)
            is_diagonal = abs(dx) > 0.1 and abs(dy) > 0.1
            
            for i in range(1, n_steps + 1):
                t = min(1.0, i / n_steps)
                xi = x0 + t * dx
                yi = y0 + t * dy
                
                # If diagonal, shift 10cm toward origin
                if is_diagonal:
                    xi, yi = self._shift_toward_origin(xi, yi, 0.1)
                
                add_point(xi, yi, z, travel_yaw)
            # set yaw at the end point for subsequent leg
            seq[-1] = HullWaypoint(point=seq[-1].point, yaw=end_yaw)

        # ---- Pass 1 at z1 = -0.3 ---------------------------------------
        # Start at (0,0), facing +x (yaw 0)
        add_point(0.0, 0.0, z1, 0.0)

        # (0,0)->(5,0) then yaw 1.57
        add_line(0.0, 0.0, 5.0, 0.0, z1, travel_yaw=0.0, end_yaw=1.57)

        # (5,0)->(5,5) then yaw 1.57
        add_line(5.0, 0.0, 5.0, 5.0, z1, travel_yaw=1.57, end_yaw=1.57)

        # (5,5)->(10,5) then yaw 1.57
        add_line(5.0, 5.0, 10.0, 5.0, z1, travel_yaw=1.57, end_yaw=1.57)

        # (10,5)->(11,5.5) then yaw 2.0 - DIAGONAL, will be shifted
        add_line(10.0, 5.0, 11.0, 5.5, z1, travel_yaw=1.57, end_yaw=2.0)

        # (11,5.5)->(12,6) then yaw 2.5 - DIAGONAL, will be shifted
        add_line(11.0, 5.5, 12.0, 6.0, z1, travel_yaw=2.0, end_yaw=2.5)

        # (12,6)->(13,6.5) then yaw 2.5 - DIAGONAL, will be shifted
        add_line(12.0, 6.0, 13.0, 6.5, z1, travel_yaw=2.5, end_yaw=2.5)

        # (13,6.5)->(13,7.5) then yaw pi
        add_line(13.0, 6.5, 13.0, 7.5, z1, travel_yaw=2.5, end_yaw=3.14159265)

        # (13,7.5)->(13.5,8.5) then yaw pi - DIAGONAL, will be shifted
        add_line(13.0, 7.5, 13.5, 8.5, z1, travel_yaw=3.14159265, end_yaw=3.14159265)

        # (13.5,8.5)->(13.5,9.5) then yaw -2.5
        add_line(13.5, 8.5, 13.5, 9.5, z1, travel_yaw=3.14159265, end_yaw=-2.5)

        # (13.5,9.5)->(12,11.5) then yaw -1.57 - DIAGONAL, will be shifted
        add_line(13.5, 9.5, 12.0, 11.5, z1, travel_yaw=-2.5, end_yaw=-1.57)

        # (12,11.5)->(11,12.5) then yaw -1.57 - DIAGONAL, will be shifted
        add_line(12.0, 11.5, 11.0, 12.5, z1, travel_yaw=-1.57, end_yaw=-1.57)

        # (11,12.5)->(4,12.5) then yaw -0.8
        add_line(11.0, 12.5, 4.0, 12.5, z1, travel_yaw=-1.57, end_yaw=-0.8)

        # (4,12.5)->(3,12) then yaw -0.6 - DIAGONAL, will be shifted
        add_line(4.0, 12.5, 3.0, 12.0, z1, travel_yaw=-0.8, end_yaw=-0.6)

        # (3,12)->(2,11) then yaw -0.6 - DIAGONAL, will be shifted
        add_line(3.0, 12.0, 2.0, 11.0, z1, travel_yaw=-0.6, end_yaw=-0.6)

        # (2,11)->(1,10) then yaw 0.0 - DIAGONAL, will be shifted
        add_line(2.0, 11.0, 1.0, 10.0, z1, travel_yaw=-0.6, end_yaw=0.0)

        # (1,10)->(0.5,8.5) then yaw 0.5 - DIAGONAL, will be shifted
        add_line(1.0, 10.0, 0.5, 8.5, z1, travel_yaw=0.0, end_yaw=0.5)

        # (0.5,8.5)->(1.5,7) then yaw 0.8 - DIAGONAL, will be shifted
        add_line(0.5, 8.5, 1.5, 7.0, z1, travel_yaw=0.5, end_yaw=0.8)

        # (1.5,7)->(3,5.5) then yaw 1.57 - DIAGONAL, will be shifted
        add_line(1.5, 7.0, 3.0, 5.5, z1, travel_yaw=0.8, end_yaw=1.57)

        # (3,5.5)->(4,5.5) then yaw 1.57
        add_line(3.0, 5.5, 4.0, 5.5, z1, travel_yaw=1.57, end_yaw=1.57)

        # (4,5.5)->(5,5) then yaw 1.57 - DIAGONAL, will be shifted
        add_line(4.0, 5.5, 5.0, 5.0, z1, travel_yaw=1.57, end_yaw=1.57)

        # ---- DESCENT at (5,5) to z2 = -1.3 ------------------------------
        add_point(5.0, 5.0, z2, 1.57)

        # ---- Pass 2 at z2 = -1.3, nudged 0.1m toward center (6.75, 8.5) --
        # Find where (5,5) first appears (after origin)
        pattern_start_idx = None
        for i, wp in enumerate(seq):
            if i > 0 and abs(wp.point[0] - 5.0) < 0.01 and abs(wp.point[1] - 5.0) < 0.01 and abs(wp.point[2] - z1) < 0.01:
                pattern_start_idx = i
                break
        
        if pattern_start_idx is None:
            self.logger.error("Could not find (5,5) waypoint in sequence!")
            return seq
        
        # Repeat from (5,5) onwards, nudging each point toward center
        for i in range(pattern_start_idx, len(seq) - 1):  # -1 to exclude the descent waypoint we just added
            wp = seq[i]
            x, y, _ = wp.point
            nx, ny = self._nudge_toward_center(x, y, self.CENTER_X, self.CENTER_Y, self.NUDGE_TOWARD_CENTER)
            add_point(nx, ny, z2, wp.yaw)
        
        self.logger.info(f"Built {len(seq)} waypoints (diagonals shifted 10cm toward origin)")
        return seq

    # --- State machine ---

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

            if self.state == MissionState.INIT:
                self.logger.info("Hull scanner mission starting.")
                self.state = MissionState.SET_AUV_MODE

            elif self.state == MissionState.SET_AUV_MODE:
                result = send_goal(self.node, self.set_target_mode, self.go_auv_goal)
                if result == SendGoalResult.SUCCESS:
                    self.state = MissionState.MOVE_TO_ORIGIN
                elif result == SendGoalResult.CANCELED:
                    self.logger.info("Mode change to AUV canceled.")
                    self.state = MissionState.CANCELED
                else:
                    self.logger.error("Failed to switch to AUV mode.")
                    self.state = MissionState.ERROR

            elif self.state == MissionState.MOVE_TO_ORIGIN:
                if self.waypoint_pub.get_subscription_count() == 0:
                    if not self._waiting_for_lqr:
                        self.logger.info("Waiting for LQR controller...")
                        self._waiting_for_lqr = True
                    continue
                self._waiting_for_lqr = False
                self.hover_start_time = None
                self.yaw_filter.reset()
                self._set_target((0.0, 0.0, self.ORIGIN_DEPTH), 0.0)
                self.state = MissionState.CHECK_ORIGIN

            elif self.state == MissionState.CHECK_ORIGIN:
                if not self.pose_available:
                    if not self._pose_missing_logged:
                        self.logger.info("Waiting for odometry...")
                        self._pose_missing_logged = True
                    continue

                distance = self._distance_to_point((0.0, 0.0, self.ORIGIN_DEPTH))
                if distance is None:
                    continue

                if distance <= self.POSITION_TOLERANCE:
                    current_yaw = self._current_yaw()
                    if current_yaw is None:
                        continue
                    yaw_error = abs(self._angular_difference(0.0, current_yaw))

                    if yaw_error > self.YAW_TOLERANCE:
                        now = self._current_time()
                        if self.last_log_time is None or now - self.last_log_time >= 1.0:
                            self.logger.info(f"Origin yaw error {math.degrees(yaw_error):.1f}°")
                            self.last_log_time = now
                        self.hover_start_time = None
                        continue

                    now = self._current_time()
                    if self.hover_start_time is None:
                        self.hover_start_time = now
                        self.logger.info("Origin reached, hovering...")
                    elif now - self.hover_start_time >= self.ORIGIN_HOVER_TIME:
                        self.logger.info("Starting hull scan.")
                        self.waypoint_index = 1  # Skip origin, already there
                        self.hover_start_time = None
                        self.last_log_time = None
                        self.state = MissionState.MOVE_TO_WAYPOINT
                else:
                    now = self._current_time()
                    if self.last_log_time is None or now - self.last_log_time >= 1.0:
                        self.logger.info(f"Moving to origin: {distance:.2f} m")
                        self.last_log_time = now
                    self.state = MissionState.MOVE_TO_ORIGIN

            elif self.state == MissionState.MOVE_TO_WAYPOINT:
                if self.waypoint_index >= len(self.hull_points):
                    self.logger.info("All waypoints complete! Mission ending at current position.")
                    self.state = MissionState.SET_ROV_MODE
                    continue

                wp = self.hull_points[self.waypoint_index]
                self._set_target(wp.point, wp.yaw)
                self.hover_start_time = None
                self.last_log_time = None
                self.state = MissionState.CHECK_WAYPOINT

            elif self.state == MissionState.CHECK_WAYPOINT:
                if not self.pose_available:
                    continue

                wp = self.hull_points[self.waypoint_index]
                distance = self._distance_to_point(wp.point)
                if distance is None:
                    continue

                if distance <= self.POSITION_TOLERANCE:
                    now = self._current_time()
                    if self.hover_start_time is None:
                        self.hover_start_time = now
                        self.logger.info(f"WP {self.waypoint_index}/{len(self.hull_points)} reached, hovering...")
                    elif now - self.hover_start_time >= self.WAYPOINT_HOVER_TIME:
                        self.waypoint_index += 1
                        self.hover_start_time = None
                        self.state = MissionState.MOVE_TO_WAYPOINT
                else:
                    now = self._current_time()
                    if self.last_log_time is None or now - self.last_log_time >= 2.0:
                        self.logger.info(f"WP {self.waypoint_index}/{len(self.hull_points)}: {distance:.2f} m")
                        self.last_log_time = now
                    self.hover_start_time = None

            elif self.state == MissionState.SET_ROV_MODE:
                self.logger.info("Switching to ROV mode at final position.")
                result = send_goal(self.node, self.set_target_mode, self.go_rov_goal)
                if result == SendGoalResult.SUCCESS:
                    self.state = MissionState.COMPLETE
                elif result == SendGoalResult.CANCELED:
                    self.logger.info("ROV mode change canceled.")
                    self.state = MissionState.CANCELED
                else:
                    self.logger.error("Failed to switch to ROV mode.")
                    self.state = MissionState.ERROR

            elif self.state == MissionState.COMPLETE:
                self.logger.info("Hull scanner mission complete!")
                break

            elif self.state == MissionState.CANCELED:
                self.logger.info("Hull scanner mission canceled.")
                break

            elif self.state == MissionState.ERROR:
                self.logger.error("Hull scanner mission error.")
                break

    def shutdown(self):
        if self.set_target_mode is not None:
            self.set_target_mode.destroy()
        if self.node is not None:
            self.node.destroy_node()


def main():
    rclpy.init()
    mission = HullScannerMission()
    try:
        mission.run()
    except KeyboardInterrupt:
        mission.logger.info('Mission interrupted.')
    finally:
        mission.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
