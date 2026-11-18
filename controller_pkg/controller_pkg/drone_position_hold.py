#!/usr/bin/env python3
"""Drone position hold (station keeping) - a test bench for tuning the
horizontal control in wind.

Climbs to <hold_altitude>, waits for the drone to settle, takes a snapshot of
the ground under it as the reference, then tries to stay exactly there.

Position is measured from the down camera. The optical-flow velocity carries
the offset from frame to frame, and the centre of the reference snapshot is
found in the current frame by template matching, which pulls the offset back
to the true offset from the hold point so it doesn't drift. Over a straight
road the match only pins down the position across the road, so each axis
uses the snapshot only when the match is sharp along that axis.
Control per horizontal axis: PID on the offset, damping from the measured
velocity, plus the wind feedforward from the virtual wind sensor.

Start it with the drone on the ground (the altimeter is zeroed there).

Topics used: /clock, /down_camera/image_raw, /altimeter, /imu (sub),
             /cmd_vel, /drone/hold_offset, /drone/wind (pub).
/drone/hold_offset (geometry_msgs/Vector3Stamped): estimated offset from the
hold point in m (x forward, y left).
"""
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, Vector3Stamped
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from ros_gz_interfaces.msg import Altimeter
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, Imu

from controller_pkg.lane_follower import PID
from controller_pkg.optical_flow import GroundVelocityEstimator
from controller_pkg.wind_estimator import WindEstimator

# Down camera mounting height above the chassis origin (drone.urdf.xacro)
DOWN_CAMERA_HEIGHT_OFFSET = 0.01


def clamp(value, limit):
    return max(-limit, min(limit, value))


class DronePositionHold(Node):

    # Fraction of the (half-resolution) image used as the reference template.
    # The template can be found for offsets up to (1 - fraction) / 2 of the image.
    TEMPLATE_FRACTION = 0.4
    # Template match score (normalized cross-correlation) below this is not trusted
    MIN_MATCH_SCORE = 0.6
    # Per-axis sharpness: score drop SHARPNESS_STEP_PX (half-res) either side of
    # the peak. Below MIN_SHARPNESS that axis is ambiguous (e.g. along a road).
    SHARPNESS_STEP_PX = 4
    MIN_SHARPNESS = 0.05
    # How far each trusted snapshot match pulls the offset (0..1 per frame)
    SNAPSHOT_BLEND = 0.5

    def __init__(self):
        super().__init__('drone_position_hold')

        self.declare_parameter('hold_altitude', 0.3)        # m above the ground
        self.declare_parameter('settle_time', 2.0)          # s at altitude before the snapshot
        self.declare_parameter('down_camera_fov_deg', 134.0)
        self.declare_parameter('kp', 8.0)                   # N per m of offset
        self.declare_parameter('ki', 2.0)                   # N per m*s
        self.declare_parameter('kd', 4.0)                   # N per m/s of velocity
        self.declare_parameter('altitude_kp', 3.0)
        self.declare_parameter('altitude_ki', 0.5)
        self.declare_parameter('altitude_kd', 2.0)
        self.declare_parameter('max_horizontal_force', 4.0)
        self.declare_parameter('max_total_horizontal_force', 8.0)
        self.declare_parameter('max_vertical_force', 3.0)
        self.declare_parameter('wind_feedforward', True)
        self.declare_parameter('wind_ff_gain', 1.0)
        self.declare_parameter('wind_filter_tau', 0.3)
        self.declare_parameter('drone_mass', 1.03)
        self.declare_parameter('air_drag_coeff', 1.0)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.hold_altitude = float(p('hold_altitude'))
        self.settle_time = float(p('settle_time'))
        self.down_fov = math.radians(float(p('down_camera_fov_deg')))
        self.kp = float(p('kp'))
        self.ki = float(p('ki'))
        self.kd = float(p('kd'))
        self.max_horizontal_force = float(p('max_horizontal_force'))
        self.max_total_horizontal_force = float(p('max_total_horizontal_force'))
        self.max_vertical_force = float(p('max_vertical_force'))
        self.wind_feedforward = bool(p('wind_feedforward'))
        self.wind_ff_gain = float(p('wind_ff_gain'))

        self.altitude_pid = PID(float(p('altitude_kp')), float(p('altitude_ki')),
                                float(p('altitude_kd')), integral_limit=2.0)
        self.wind = WindEstimator(float(p('drone_mass')), float(p('air_drag_coeff')),
                                  float(p('wind_filter_tau')))
        self.velocity = None  # GroundVelocityEstimator, created on the first frame
        self.bridge = CvBridge()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.offset_pub = self.create_publisher(Vector3Stamped, '/drone/hold_offset', 1)
        self.wind_pub = self.create_publisher(Vector3Stamped, '/drone/wind', 1)
        self.create_subscription(Clock, '/clock', self.clock_callback, 1)
        self.create_subscription(Altimeter, '/altimeter', self.altimeter_callback, 1)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Image, '/down_camera/image_raw', self.image_callback, 1)

        self.sim_time = None
        self.state = 'WAIT_CLOCK'
        self.state_start_time = None
        self.last_control_time = None
        self.altitude = None
        self.ground_level = None

        self.reference = None            # grayscale snapshot of the hold point
        self.template_origin = (0, 0)    # top-left of the template in the snapshot
        self.offset = (0.0, 0.0)         # (forward, left) m from the hold point
        self.offset_source = '--'
        self.ground_velocity = (0.0, 0.0)
        self.integral = [0.0, 0.0]       # N, (forward, left)
        self.last_image_time = None
        self.horizontal_cmd = (0.0, 0.0)  # (forward, left) N, PID part
        self.last_cmd = (0.0, 0.0, 0.0)   # (forward, left, up) N sent on /cmd_vel

        self.create_timer(0.05, self.control_loop)  # 20 Hz
        self.get_logger().info('Drone position hold started, waiting for /clock...')

    # ---------------------------------------------------------------- sensors

    def clock_callback(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec / 1e9

    def altimeter_callback(self, msg):
        if self.ground_level is None:
            self.ground_level = msg.vertical_position
        self.altitude = msg.vertical_position - self.ground_level

    def flying(self):
        return self.altitude is not None and self.altitude > 0.1 and self.state in ('CLIMB', 'HOLD')

    def imu_callback(self, msg):
        if not self.flying():
            return
        a = msg.linear_acceleration
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.wind.update((a.x, a.y, a.z), self.last_cmd, stamp)

    def gray(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Half resolution; the area averaging also smooths the camera noise
        return cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        dt = 0.0 if self.last_image_time is None else stamp - self.last_image_time
        self.last_image_time = stamp
        if self.velocity is None:
            self.velocity = GroundVelocityEstimator(self.down_fov, frame.shape[1])

        height = None if self.altitude is None else self.altitude + DOWN_CAMERA_HEIGHT_OFFSET
        if not self.flying() or height is None:
            return
        velocity = self.velocity.update(frame, stamp, height)
        if velocity is not None:
            self.ground_velocity = velocity
        metres_per_px = 2.0 * height * math.tan(self.down_fov / 2.0) / frame.shape[1]

        if self.state == 'HOLD':
            if self.reference is None:
                # Snapshot of the hold point
                gray = self.gray(frame)
                h, w = gray.shape
                th, tw = int(h * self.TEMPLATE_FRACTION), int(w * self.TEMPLATE_FRACTION)
                self.template_origin = ((w - tw) // 2, (h - th) // 2)
                ox, oy = self.template_origin
                self.reference = gray[oy:oy + th, ox:ox + tw].copy()
                self.offset = (0.0, 0.0)
                self.integral = [0.0, 0.0]
                self.get_logger().info('Hold reference snapshot taken')
            self.update_offset(frame, metres_per_px, dt)

        self.update_horizontal_command(dt)

    def update_offset(self, frame, metres_per_px, dt):
        """Offset from the hold point. The optical-flow velocity carries the
        offset from frame to frame; on each axis where the reference snapshot
        is matched unambiguously, the offset is pulled toward that match."""
        # Dead reckoning with the measured ground velocity
        v_forward, v_left = self.ground_velocity
        offset = [self.offset[0] + v_forward * dt, self.offset[1] + v_left * dt]

        gray = self.gray(frame)
        scores = cv2.matchTemplate(gray, self.reference, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        dx, dy = x - self.template_origin[0], y - self.template_origin[1]

        # Over a straight road the match is only sharp across the road: along it
        # the score barely changes (aperture problem). Per image axis, the drop
        # in score a few pixels either side of the peak says whether it's sharp.
        k = self.SHARPNESS_STEP_PX
        sh, sw = scores.shape
        sharp_x = score - max(scores[y, max(x - k, 0)], scores[y, min(x + k, sw - 1)])
        sharp_y = score - max(scores[max(y - k, 0), x], scores[min(y + k, sh - 1), x])

        # The ground moves down the image when the drone moves forward and
        # right in the image when it moves left (see optical_flow.py)
        scale = metres_per_px / 0.5  # half-resolution pixels
        used = []
        if score >= self.MIN_MATCH_SCORE:
            if sharp_y >= self.MIN_SHARPNESS:
                offset[0] += self.SNAPSHOT_BLEND * (dy * scale - offset[0])
                used.append('fwd')
            if sharp_x >= self.MIN_SHARPNESS:
                offset[1] += self.SNAPSHOT_BLEND * (dx * scale - offset[1])
                used.append('left')
        self.offset = tuple(offset)
        self.offset_source = (f'snapshot used for: {"/".join(used) or "none"}; '
                              f'score {score:.2f} sharpness fwd {sharp_y:.2f} left {sharp_x:.2f}')

        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'chassis'
        msg.vector.x, msg.vector.y = self.offset
        self.offset_pub.publish(msg)

        wind = self.wind.wind_velocity(self.ground_velocity)
        if wind is not None:
            wmsg = Vector3Stamped()
            wmsg.header = msg.header
            wmsg.vector.x, wmsg.vector.y = wind
            self.wind_pub.publish(wmsg)

    def update_horizontal_command(self, dt):
        forces = []
        for i in range(2):
            error = -self.offset[i] if self.state == 'HOLD' else 0.0
            if self.state == 'HOLD':
                self.integral[i] = clamp(self.integral[i] + self.ki * error * dt,
                                         self.max_horizontal_force)
            force = self.kp * error + self.integral[i] - self.kd * self.ground_velocity[i]
            forces.append(clamp(force, self.max_horizontal_force))
        self.horizontal_cmd = tuple(forces)

    # ---------------------------------------------------------------- control

    def set_state(self, state):
        self.state = state
        self.state_start_time = self.sim_time
        self.get_logger().info(f'State -> {state}')

    def elapsed(self):
        return self.sim_time - self.state_start_time

    def publish_cmd(self, forward, left, up):
        if self.wind_feedforward and self.flying():
            ff_forward, ff_left = self.wind.feedforward(self.ground_velocity)
            forward += self.wind_ff_gain * ff_forward
            left += self.wind_ff_gain * ff_left
        forward = clamp(forward, self.max_total_horizontal_force)
        left = clamp(left, self.max_total_horizontal_force)
        self.last_cmd = (forward, left, up)
        twist = Twist()
        twist.linear.x = forward
        twist.linear.y = left
        twist.linear.z = up
        self.cmd_pub.publish(twist)

    def control_loop(self):
        if self.sim_time is None:
            return
        dt = 0.0 if self.last_control_time is None else self.sim_time - self.last_control_time
        self.last_control_time = self.sim_time

        if self.state == 'WAIT_CLOCK':
            self.set_state('REGISTER_DELAY')
        elif self.state == 'REGISTER_DELAY':
            if self.elapsed() >= 1.0 and self.altitude is not None:
                self.altitude_pid.reset()
                self.set_state('CLIMB')
        elif self.state in ('CLIMB', 'HOLD'):
            up = clamp(self.altitude_pid.update(self.hold_altitude - self.altitude, dt),
                       self.max_vertical_force)
            self.publish_cmd(self.horizontal_cmd[0], self.horizontal_cmd[1], up)
            if (self.state == 'CLIMB' and self.elapsed() > self.settle_time
                    and abs(self.hold_altitude - self.altitude) < 0.03):
                self.set_state('HOLD')
            if self.state == 'HOLD':
                self.get_logger().info(
                    f'alt {self.altitude:.3f} offset fwd {self.offset[0]:+.3f} left {self.offset[1]:+.3f} m '
                    f'({self.offset_source})', throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = DronePositionHold()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        twist = Twist()
        node.cmd_pub.publish(twist)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
