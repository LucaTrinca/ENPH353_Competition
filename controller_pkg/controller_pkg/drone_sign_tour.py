#!/usr/bin/env python3
"""Drone sign tour: fly a hardcoded route to a viewing point in front of every
clue board, using dead reckoning only.

The route comes from course_map.sign_tour(): for each sign, fly to a point
0.35 m in front of its banner at 0.4 m, turn to face it, hold, and save a
front camera photo (the front camera is tilted down to see the low banners).
For the mountain sign it climbs to 2.3 m on the spot first and crosses high.
Random signs are aimed at the middle of their spawn range.

Position is tracked from the known start pose (course_map.START), with no
absolute sensing:
  - yaw: integrated IMU gyro
  - velocity: predicted with the IMU accelerometer at 100 Hz, corrected with
    the optical-flow ground velocity from the centre of the down camera (not
    while turning fast, since the flow doesn't handle rotation, and not over
    featureless ground such as open water)
  - position: integrated velocity
Height above the ground comes from the downward range sensor, with the
altimeter as a fallback; 'abs' route steps hold an altimeter height instead.

Control works in accelerations (m/s^2), sent as forces (x drone_mass) on
/cmd_vel: a carrot moves along each leg at cruise speed and a PD (+ small I)
on position pulls the drone after it.

Start it with the drone on the ground at its spawn point.

Topics used: /clock, /imu, /down_camera/image_raw, /down_depth, /altimeter,
             /camera/image_raw (sub), /cmd_vel, /score_tracker,
             /drone/pose_estimate (pub).
"""
import math
import os
from datetime import datetime

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from ros_gz_interfaces.msg import Altimeter
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String

from controller_pkg.course_map import START, sign_tour
from controller_pkg.imu_utils import level_acceleration
from controller_pkg.lane_follower import PID
from controller_pkg.optical_flow import GroundVelocityEstimator

# Down camera mounting height above the chassis origin (drone.urdf.xacro)
DOWN_CAMERA_HEIGHT_OFFSET = 0.01


def clamp(value, limit):
    return max(-limit, min(limit, value))


def wrap(angle):
    return math.remainder(angle, 2.0 * math.pi)


class DroneSignTour(Node):

    def __init__(self):
        super().__init__('drone_sign_tour')

        # Team ID and password: max 8 characters each, no spaces or commas
        self.declare_parameter('team_id', 'TeamFizz')
        self.declare_parameter('team_password', 'pass1234')
        self.declare_parameter('drone_mass', 20.0)          # kg, must match the URDF
        self.declare_parameter('down_camera_fov_deg', 134.0)
        self.declare_parameter('cruise_speed', 0.4)         # m/s along each leg
        self.declare_parameter('view_distance', 0.35)       # m in front of each banner
        self.declare_parameter('transit_altitude', 0.4)     # m above the ground
        self.declare_parameter('view_altitude', 0.4)        # m above the ground
        self.declare_parameter('mountain_altitude', 2.3)    # m above the start (altimeter)
        self.declare_parameter('dwell', 2.0)                # s held at each sign
        # Gains and limits are accelerations (m/s^2 per unit of error)
        self.declare_parameter('position_kp', 3.0)          # per m
        self.declare_parameter('position_ki', 0.5)          # per m*s
        self.declare_parameter('position_kd', 4.0)          # per m/s
        self.declare_parameter('max_horizontal_accel', 2.0)
        self.declare_parameter('altitude_kp', 3.0)
        self.declare_parameter('altitude_ki', 0.5)
        self.declare_parameter('altitude_kd', 3.0)
        self.declare_parameter('max_vertical_accel', 3.0)
        self.declare_parameter('max_climb_rate', 0.3)       # m/s, the altitude target ramps at this rate
        self.declare_parameter('yaw_kp', 2.0)               # rad/s per rad
        self.declare_parameter('max_yaw_rate', 0.8)         # rad/s
        self.declare_parameter('arrive_tolerance', 0.12)    # m
        self.declare_parameter('yaw_tolerance', 0.1)        # rad
        self.declare_parameter('step_timeout', 8.0)         # s beyond the leg's travel time
        self.declare_parameter('flow_gain', 0.3)            # weight of each flow correction
        self.declare_parameter('flow_max_yaw_rate', 0.2)    # rad/s, no flow corrections above this
        self.declare_parameter('flow_max_vertical_speed', 0.15)  # m/s, no flow corrections above this
        self.declare_parameter('flow_crop', 0.5)            # central fraction of the down image used
        self.declare_parameter('flow_min_texture', 6.0)     # grey-level std, below = nothing to track
        self.declare_parameter('range_timeout', 0.3)        # s
        self.declare_parameter('photo_dir', '~/drone_images')
        self.declare_parameter('show_front_camera', True)   # live front camera window

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.team_id = p('team_id')
        self.team_password = p('team_password')
        self.mass = float(p('drone_mass'))
        self.down_fov = math.radians(float(p('down_camera_fov_deg')))
        self.cruise_speed = float(p('cruise_speed'))
        self.position_kp = float(p('position_kp'))
        self.position_ki = float(p('position_ki'))
        self.position_kd = float(p('position_kd'))
        self.max_horizontal_accel = float(p('max_horizontal_accel'))
        self.max_vertical_accel = float(p('max_vertical_accel'))
        self.max_climb_rate = float(p('max_climb_rate'))
        self.yaw_kp = float(p('yaw_kp'))
        self.max_yaw_rate = float(p('max_yaw_rate'))
        self.arrive_tolerance = float(p('arrive_tolerance'))
        self.yaw_tolerance = float(p('yaw_tolerance'))
        self.step_timeout = float(p('step_timeout'))
        self.flow_gain = float(p('flow_gain'))
        self.flow_max_yaw_rate = float(p('flow_max_yaw_rate'))
        self.flow_max_vertical_speed = float(p('flow_max_vertical_speed'))
        self.flow_crop = float(p('flow_crop'))
        self.flow_min_texture = float(p('flow_min_texture'))
        self.range_timeout = float(p('range_timeout'))
        self.show_front_camera = bool(p('show_front_camera'))
        self.photo_dir = os.path.join(os.path.expanduser(p('photo_dir')),
                                      datetime.now().strftime('tour_%Y%m%d_%H%M%S'))

        self.steps = sign_tour(float(p('view_distance')), float(p('transit_altitude')),
                               float(p('view_altitude')), float(p('mountain_altitude')),
                               float(p('dwell')))
        self.altitude_pid = PID(float(p('altitude_kp')), float(p('altitude_ki')),
                                float(p('altitude_kd')), integral_limit=2.0)
        self.velocity_estimator = None
        self.bridge = CvBridge()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.score_pub = self.create_publisher(String, '/score_tracker', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/drone/pose_estimate', 1)
        self.create_subscription(Clock, '/clock', self.clock_callback, 1)
        self.create_subscription(Imu, '/imu', self.imu_callback, 20)
        self.create_subscription(Altimeter, '/altimeter', self.altimeter_callback, 1)
        self.create_subscription(Image, '/down_depth', self.range_callback, 1)
        self.create_subscription(Image, '/down_camera/image_raw', self.down_image_callback, 1)
        self.create_subscription(Image, '/camera/image_raw', self.front_image_callback, 1)

        self.sim_time = None
        self.state = 'WAIT_CLOCK'
        self.state_start_time = None
        self.last_control_time = None

        # Height sensing
        self.altitude = None        # altimeter, zeroed on the ground at startup
        self.vertical_speed = 0.0   # altimeter, m/s
        self.ground_level = None
        self.range_height = None    # downward range sensor
        self.range_time = None

        # Dead-reckoned state in the course frame
        self.x, self.y, self.yaw = START
        self.vx = self.vy = 0.0
        self.yaw_rate = 0.0
        self.last_imu_time = None

        # Route progress
        self.step_index = -1
        self.step_start_time = None
        self.leg_start = (self.x, self.y)
        self.arrived_time = None
        self.integral = [0.0, 0.0]
        self.altitude_setpoint = 0.0
        self.step_climb_time = 0.0
        self.front_frame = None
        if self.show_front_camera:
            cv2.namedWindow('Front Camera', cv2.WINDOW_NORMAL)

        self.create_timer(0.05, self.control_loop)  # 20 Hz
        self.get_logger().info(f'Drone sign tour started ({len(self.steps)} route steps), '
                               'waiting for /clock...')

    # ---------------------------------------------------------------- sensors

    def clock_callback(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec / 1e9

    def altimeter_callback(self, msg):
        if self.ground_level is None:
            self.ground_level = msg.vertical_position
        self.altitude = msg.vertical_position - self.ground_level
        self.vertical_speed = msg.vertical_velocity

    def range_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        valid = depth[np.isfinite(depth)]
        if valid.size:
            self.range_height = float(np.median(valid))
            self.range_time = self.sim_time

    def height_above_ground(self):
        if (self.range_height is not None and self.range_time is not None
                and self.sim_time is not None
                and self.sim_time - self.range_time <= self.range_timeout):
            return self.range_height
        return self.altitude

    def flying(self):
        h = self.height_above_ground()
        return h is not None and h > 0.05 and self.state == 'TOUR'

    def imu_callback(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        dt = 0.0 if self.last_imu_time is None else stamp - self.last_imu_time
        self.last_imu_time = stamp
        if dt <= 0.0 or dt > 0.1:
            return
        self.yaw_rate = msg.angular_velocity.z
        if not self.flying():
            self.vx = self.vy = 0.0
            return
        self.yaw = wrap(self.yaw + self.yaw_rate * dt)
        # Horizontal acceleration, body -> course frame (the drone is kept level)
        ax, ay = level_acceleration(msg)  # gravity leak from tilt removed
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.vx += (c * ax - s * ay) * dt
        self.vy += (s * ax + c * ay) * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def down_image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.velocity_estimator is None:
            self.velocity_estimator = GroundVelocityEstimator(
                self.down_fov, frame.shape[1], smoothing=0.3,
                crop_fraction=self.flow_crop, min_texture=self.flow_min_texture)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        h = self.height_above_ground()
        if not self.flying() or h is None:
            self.velocity_estimator.reset()
            return
        velocity = self.velocity_estimator.update(frame, stamp, h + DOWN_CAMERA_HEIGHT_OFFSET)
        # Turning rotates the image and climbing zooms it; the flow assumes neither
        if (velocity is None or abs(self.yaw_rate) > self.flow_max_yaw_rate
                or abs(self.vertical_speed) > self.flow_max_vertical_speed):
            return
        # Correct the IMU-predicted velocity with the measured ground velocity
        forward, left = velocity
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.vx += self.flow_gain * ((c * forward - s * left) - self.vx)
        self.vy += self.flow_gain * ((s * forward + c * left) - self.vy)

    def front_image_callback(self, msg):
        self.front_frame = msg
        if not self.show_front_camera:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8').copy()
        if 0 <= self.step_index < len(self.steps):
            status = f'{self.step_index + 1}/{len(self.steps)} {self.steps[self.step_index][0]}'
        else:
            status = self.state
        estimate = f'est x {self.x:+.2f} y {self.y:+.2f} yaw {math.degrees(self.yaw):+.0f}'
        for row, text in enumerate([status, estimate]):
            y = 25 + 25 * row
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow('Front Camera', frame)
        cv2.waitKey(1)

    # ---------------------------------------------------------------- control

    def send_score(self, location, prediction):
        msg = String()
        msg.data = f'{self.team_id},{self.team_password},{location},{prediction}'
        self.score_pub.publish(msg)
        self.get_logger().info(f'Sent to /score_tracker: {msg.data}')

    def set_state(self, state):
        self.state = state
        self.state_start_time = self.sim_time
        self.get_logger().info(f'State -> {state}')

    def publish_accel(self, a_x, a_y, a_up, yaw_rate):
        """Course-frame horizontal acceleration -> body-frame force on /cmd_vel."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        twist = Twist()
        twist.linear.x = self.mass * (c * a_x + s * a_y)
        twist.linear.y = self.mass * (-s * a_x + c * a_y)
        twist.linear.z = self.mass * a_up
        twist.angular.z = yaw_rate
        self.cmd_pub.publish(twist)

    def start_step(self, index):
        self.step_index = index
        self.step_start_time = self.sim_time
        self.leg_start = (self.x, self.y)
        self.arrived_time = None
        self.altitude_pid.prev_error = None  # the altitude reference may change mode
        # The altitude target ramps from the current height (in this step's mode)
        mode = self.steps[index][4]
        self.altitude_setpoint = self.altitude if mode == 'abs' else self.height_above_ground()
        self.step_climb_time = abs(self.steps[index][3] - self.altitude_setpoint) / self.max_climb_rate
        label = self.steps[index][0]
        self.get_logger().info(f'Step {index + 1}/{len(self.steps)}: {label} '
                               f'(estimate x {self.x:+.2f} y {self.y:+.2f} '
                               f'yaw {math.degrees(self.yaw):+.0f} deg)')

    def save_photo(self, label):
        if self.front_frame is None:
            self.get_logger().warn(f'No front camera image for {label}')
            return
        os.makedirs(self.photo_dir, exist_ok=True)
        path = os.path.join(self.photo_dir, label.replace(' ', '_') + '.png')
        cv2.imwrite(path, self.bridge.imgmsg_to_cv2(self.front_frame, desired_encoding='bgr8'))
        self.get_logger().info(f'Photo {path} (estimate x {self.x:+.2f} y {self.y:+.2f} '
                               f'yaw {math.degrees(self.yaw):+.0f} deg)')

    def tour_step(self, dt):
        label, tx, ty, altitude, mode, yaw_target, dwell, photo = self.steps[self.step_index]
        elapsed = self.sim_time - self.step_start_time

        # Carrot moving from the leg start toward the target at cruise speed
        sx, sy = self.leg_start
        length = math.hypot(tx - sx, ty - sy)
        travelled = min(length, self.cruise_speed * elapsed)
        ux, uy = ((tx - sx) / length, (ty - sy) / length) if length > 1e-6 else (0.0, 0.0)
        cx, cy = sx + ux * travelled, sy + uy * travelled
        carrot_speed = self.cruise_speed if travelled < length else 0.0

        a = []
        for i, (error, v, v_ref) in enumerate([(cx - self.x, self.vx, ux * carrot_speed),
                                               (cy - self.y, self.vy, uy * carrot_speed)]):
            self.integral[i] = clamp(self.integral[i] + self.position_ki * error * dt,
                                     self.max_horizontal_accel)
            a.append(self.position_kp * error + self.position_kd * (v_ref - v) + self.integral[i])
        norm = math.hypot(*a)
        if norm > self.max_horizontal_accel:
            a = [ai * self.max_horizontal_accel / norm for ai in a]

        height = self.altitude if mode == 'abs' else self.height_above_ground()
        step = self.max_climb_rate * dt
        self.altitude_setpoint += clamp(altitude - self.altitude_setpoint, step)
        altitude_error = self.altitude_setpoint - height
        a_up = clamp(self.altitude_pid.update(altitude_error, dt), self.max_vertical_accel)

        yaw_error = 0.0 if yaw_target is None else wrap(yaw_target - self.yaw)
        yaw_rate = clamp(self.yaw_kp * yaw_error, self.max_yaw_rate)
        self.publish_accel(a[0], a[1], a_up, yaw_rate)

        # Arrival: at the target, at the altitude and facing the right way
        position_error = math.hypot(tx - self.x, ty - self.y)
        arrived = (travelled >= length and position_error < self.arrive_tolerance
                   and abs(altitude - height) < 0.05 and abs(yaw_error) < self.yaw_tolerance)
        if arrived and self.arrived_time is None:
            self.arrived_time = self.sim_time
        timed_out = elapsed > max(length / self.cruise_speed, self.step_climb_time) + self.step_timeout
        if timed_out and self.arrived_time is None:
            self.get_logger().warn(f'{label}: timed out (position error {position_error:.2f} m, '
                                   f'altitude error {altitude_error:+.2f} m, '
                                   f'yaw error {math.degrees(yaw_error):+.0f} deg), moving on')
            self.arrived_time = self.sim_time - dwell
        if self.arrived_time is not None and self.sim_time - self.arrived_time >= dwell:
            if photo:
                self.save_photo(label)
            return True
        return False

    def control_loop(self):
        if self.sim_time is None:
            return
        dt = 0.0 if self.last_control_time is None else self.sim_time - self.last_control_time
        self.last_control_time = self.sim_time

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'course'
        pose.pose.position.x, pose.pose.position.y = self.x, self.y
        pose.pose.orientation.z, pose.pose.orientation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        self.pose_pub.publish(pose)

        if self.state == 'WAIT_CLOCK':
            self.set_state('REGISTER_DELAY')
        elif self.state == 'REGISTER_DELAY':
            if self.sim_time - self.state_start_time >= 1.0 and self.altitude is not None:
                self.send_score(0, 'NA')  # start the timer
                self.altitude_pid.reset()
                self.set_state('TOUR')
                self.start_step(0)
        elif self.state == 'TOUR':
            if self.tour_step(dt):
                if self.step_index + 1 < len(self.steps):
                    self.start_step(self.step_index + 1)
                else:
                    self.send_score(-1, 'NA')  # stop the timer
                    self.set_state('DONE')
        elif self.state == 'DONE':
            # Hold the last step's position and altitude
            self.tour_step(dt)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = DroneSignTour()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
