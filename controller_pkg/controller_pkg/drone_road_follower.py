#!/usr/bin/env python3
"""Drone road follower.

Starts the score tracker timer, climbs to a cruise altitude, then follows the
road using the downward-facing camera while holding altitude with the
altimeter. Stops the timer after <follow_duration> sim seconds and keeps
hovering in place (the drone never lands: every ground contact is a -5 pt
penalty).

Height above the ground comes from the downward range sensor (a tiny depth
camera pointing straight down), which also follows the terrain. When it has no valid
reading (on the ground, or out of range) the altimeter is used instead. Start
this node with the drone on the ground: the first altimeter reading is taken
as ground level, so the altimeter fallback is also a height above the ground.

Wind pushes the drone around, so the horizontal control uses the drone's
velocity over the ground, measured with optical flow on the down camera
(GroundVelocityEstimator). The controllers work in accelerations (m/s^2), so
the same gains suit any drone mass; publish_cmd turns them into forces
(mass x acceleration) on /cmd_vel, adds the wind feedforward (a force, since
the wind's push doesn't depend on mass) and sends the yaw rate in rad/s:
  - altitude:  PID on the altimeter height                     -> linear.z
  - lateral:   PID on the road offset under the drone [m], with
               damping from the measured sideways velocity     -> linear.y
  - forward:   PI on the measured forward ground speed         -> linear.x
  - heading:   P on the road direction ahead                   -> angular.z
The integral terms soak up the steady push of the wind. If the road is lost
(e.g. the wind carried the drone across a hairpin) the drone flies to the
nearest road in view, or with no road in view back to where it last saw the
road (tracked with the measured velocity), and climbs slowly to widen its view.

Wind feedforward: a virtual wind sensor (WindEstimator) measures the air
force on the drone from the IMU and the force commands it sends, and turns it
into the wind velocity using the optical-flow ground velocity. The force that
cancels the wind (but not the drone's own air drag) is added to the commands
in real time, so the PID terms only have to correct what is left.

Topics used: /clock, /down_camera/image_raw, /down_depth, /altimeter, /imu (sub),
             /camera/image_raw (sub, display only, if show_front_camera),
             /cmd_vel, /score_tracker, /drone/wind (pub).
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
from std_msgs.msg import String

from controller_pkg.lane_follower import PID, DownwardRoadDetector
from controller_pkg.optical_flow import GroundVelocityEstimator
from controller_pkg.wind_estimator import WindEstimator

# Down camera mounting height above the chassis origin (drone.urdf.xacro)
DOWN_CAMERA_HEIGHT_OFFSET = 0.01


def clamp(value, limit):
    return max(-limit, min(limit, value))


class DroneRoadFollower(Node):

    def __init__(self):
        super().__init__('drone_road_follower')

        # Team ID and password: max 8 characters each, no spaces or commas
        self.declare_parameter('team_id', 'TeamFizz')
        self.declare_parameter('team_password', 'pass1234')
        self.declare_parameter('cruise_altitude', 0.3)      # m above the ground
        self.declare_parameter('max_search_altitude', 0.6)  # m, climb limit when road is lost
        self.declare_parameter('follow_duration', 60.0)     # sim s of road following
        self.declare_parameter('cruise_speed', 0.4)         # m/s over the ground
        self.declare_parameter('down_camera_fov_deg', 134.0)  # must match drone_gazebo.xacro
        # Gains and limits are accelerations (m/s^2 per unit of error), per kg of drone
        self.declare_parameter('altitude_kp', 3.0)          # m/s^2 per m of height error
        self.declare_parameter('altitude_ki', 0.5)
        self.declare_parameter('altitude_kd', 3.0)          # m/s^2 per m/s
        self.declare_parameter('lateral_kp', 10.0)          # m/s^2 per m of road offset
        self.declare_parameter('lateral_ki', 4.0)           # m/s^2 per m*s
        self.declare_parameter('lateral_kd', 5.0)           # m/s^2 per m/s of sideways velocity
        self.declare_parameter('speed_kp', 5.0)             # m/s^2 per m/s of speed error
        self.declare_parameter('speed_ki', 3.0)             # m/s^2 per m
        self.declare_parameter('yaw_kp', 1.5)               # rad/s per rad of heading error
        self.declare_parameter('max_vertical_accel', 3.0)   # m/s^2
        self.declare_parameter('max_horizontal_accel', 4.0)  # m/s^2, PID part
        self.declare_parameter('max_yaw_rate', 1.0)
        self.declare_parameter('show_debug', True)          # down camera debug window
        self.declare_parameter('show_front_camera', True)   # front camera window
        self.declare_parameter('wind_feedforward', False)   # cancel the measured wind (light drones)
        self.declare_parameter('wind_ff_gain', 1.0)         # fraction of the wind force cancelled
        self.declare_parameter('wind_filter_tau', 0.3)      # s, wind sensor low-pass filter
        self.declare_parameter('drone_mass', 20.0)          # kg, must match the URDF
        self.declare_parameter('use_range_sensor', True)    # height from the downward depth sensor
        self.declare_parameter('range_timeout', 0.3)        # s, older readings fall back to the altimeter
        self.declare_parameter('air_drag_coeff', 1.0)       # N per m/s, wind plugin scaling factor
        self.declare_parameter('max_total_horizontal_accel', 8.0)  # m/s^2, PID + feedforward

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.team_id = p('team_id')
        self.team_password = p('team_password')
        self.cruise_altitude = float(p('cruise_altitude'))
        self.max_search_altitude = float(p('max_search_altitude'))
        self.follow_duration = float(p('follow_duration'))
        self.cruise_speed = float(p('cruise_speed'))
        self.down_fov = math.radians(float(p('down_camera_fov_deg')))
        self.lateral_kp = float(p('lateral_kp'))
        self.lateral_ki = float(p('lateral_ki'))
        self.lateral_kd = float(p('lateral_kd'))
        self.speed_kp = float(p('speed_kp'))
        self.speed_ki = float(p('speed_ki'))
        self.yaw_kp = float(p('yaw_kp'))
        self.max_vertical_accel = float(p('max_vertical_accel'))
        self.max_horizontal_accel = float(p('max_horizontal_accel'))
        self.mass = float(p('drone_mass'))
        self.max_yaw_rate = float(p('max_yaw_rate'))
        self.show_debug = bool(p('show_debug'))
        self.show_front_camera = bool(p('show_front_camera'))
        self.wind_feedforward = bool(p('wind_feedforward'))
        self.use_range_sensor = bool(p('use_range_sensor'))
        self.range_timeout = float(p('range_timeout'))
        self.wind_ff_gain = float(p('wind_ff_gain'))
        self.max_total_horizontal_accel = float(p('max_total_horizontal_accel'))

        self.altitude_pid = PID(float(p('altitude_kp')), float(p('altitude_ki')),
                                float(p('altitude_kd')), integral_limit=2.0)
        self.detector = DownwardRoadDetector()
        self.wind = WindEstimator(float(p('drone_mass')), float(p('air_drag_coeff')),
                                  float(p('wind_filter_tau')))
        self.ground_velocity = (0.0, 0.0)  # (forward, left) m/s from optical flow
        self.last_cmd = (0.0, 0.0, 0.0)    # (forward, left, up) N last sent on /cmd_vel
        self.wind_ff = (0.0, 0.0)          # (forward, left) N feedforward in the last command
        self.velocity = None  # GroundVelocityEstimator, created on the first frame

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.score_pub = self.create_publisher(String, '/score_tracker', 1)
        self.create_subscription(Clock, '/clock', self.clock_callback, 1)
        self.create_subscription(Altimeter, '/altimeter', self.altimeter_callback, 1)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Image, '/down_depth', self.range_callback, 1)
        self.wind_pub = self.create_publisher(Vector3Stamped, '/drone/wind', 1)
        self.create_subscription(Image, '/down_camera/image_raw', self.image_callback, 1)
        if self.show_front_camera:
            self.create_subscription(Image, '/camera/image_raw', self.front_image_callback, 1)

        self.bridge = CvBridge()
        if self.show_debug:
            cv2.namedWindow('Down Camera', cv2.WINDOW_NORMAL)
        if self.show_front_camera:
            cv2.namedWindow('Front Camera', cv2.WINDOW_NORMAL)

        self.sim_time = None
        self.state = 'WAIT_CLOCK'
        self.state_start_time = None

        self.altitude = None      # height above the ground from the altimeter
        self.ground_level = None  # altimeter reading on the ground at startup
        self.range_height = None  # height above the ground from the downward depth sensor
        self.range_time = None
        self.target_altitude = self.cruise_altitude
        self.last_control_time = None

        # Horizontal control state, updated on every down camera frame
        self.road_visible = False
        self.last_image_time = None
        # Horizontal commands and integrals are accelerations [m/s^2]
        self.lateral_integral = 0.0   # sideways wind compensation
        self.forward_integral = 0.0   # forward wind compensation
        self.follow_cmd = (0.0, 0.0, 0.0)  # (forward, left, yaw) while following the road
        self.hold_cmd = (0.0, 0.0)         # (forward, left) to hold position
        self.return_cmd = None             # (forward, left) back to the last road seen
        # Last road point seen under the drone, (forward, right) in m from the drone
        self.last_road_point = None

        self.create_timer(0.05, self.control_loop)  # 20 Hz
        self.get_logger().info('Drone road follower started, waiting for /clock...')

    # ---------------------------------------------------------------- sensors

    def clock_callback(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec / 1e9

    def altimeter_callback(self, msg):
        # The altimeter reads height relative to the spawn point (0.1 m above the
        # ground), so zero it on the first reading while the drone is on the ground
        if self.ground_level is None:
            self.ground_level = msg.vertical_position
            self.get_logger().info(f'Ground level: altimeter {self.ground_level:+.3f} m')
        self.altitude = msg.vertical_position - self.ground_level

    def range_callback(self, msg):
        # Depth image in metres; pixels with nothing in range (e.g. closer than the
        # 0.1 m near clip while on the ground) are inf. The view is narrow
        # (0.2 rad), so the depth is the vertical distance.
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        valid = depth[np.isfinite(depth)]
        if valid.size:
            self.range_height = float(np.median(valid))
            self.range_time = self.sim_time

    def height_reading(self):
        """(height above the ground [m], source): the downward depth sensor if it
        has a fresh reading, otherwise the altimeter."""
        if (self.use_range_sensor and self.range_height is not None
                and self.range_time is not None and self.sim_time is not None
                and self.sim_time - self.range_time <= self.range_timeout):
            return self.range_height, 'range'
        return self.altitude, 'altimeter'

    def height(self):
        return self.height_reading()[0]

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.velocity is None:
            self.velocity = GroundVelocityEstimator(self.down_fov, frame.shape[1])
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.update_horizontal_commands(frame, stamp)
        if self.show_debug:
            self.show_debug_window(frame)

    def flying(self):
        return (self.height() is not None and self.height() > 0.1
                and self.state in ('CLIMB', 'FOLLOW', 'HOVER', 'DONE'))

    def imu_callback(self, msg):
        # Only while flying: on the ground the ground's push would read as wind
        if not self.flying():
            return
        a = msg.linear_acceleration
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.wind.update((a.x, a.y, a.z), self.last_cmd, stamp)

    def publish_wind(self, stamp):
        wind = self.wind.wind_velocity(self.ground_velocity)
        if wind is None:
            return
        msg = Vector3Stamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'chassis'  # x forward, y left
        msg.vector.x, msg.vector.y = wind
        self.wind_pub.publish(msg)

    def front_image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('Front Camera', frame)
        cv2.waitKey(1)

    def update_horizontal_commands(self, frame, stamp):
        dt = 0.0 if self.last_image_time is None else stamp - self.last_image_time
        self.last_image_time = stamp

        height = None if self.height() is None else self.height() + DOWN_CAMERA_HEIGHT_OFFSET
        flying = height is not None and height > 0.1
        velocity = self.velocity.update(frame, stamp, height) if flying else None
        v_forward, v_left = velocity if velocity is not None else (0.0, 0.0)
        if velocity is not None:
            self.ground_velocity = velocity
        self.publish_wind(self.get_clock().now().to_msg())
        integrate = flying and self.state in ('FOLLOW', 'HOVER', 'DONE')

        # Hold position: zero ground velocity, keeping the wind compensation
        hold_forward = self.speed_kp * (0.0 - v_forward) + self.forward_integral
        hold_left = self.lateral_kd * (0.0 - v_left) - self.lateral_integral
        self.hold_cmd = (clamp(hold_forward, self.max_horizontal_accel),
                         clamp(hold_left, self.max_horizontal_accel))

        # The road the drone came from is opposite its direction of travel
        # (straight behind when it is barely moving)
        if velocity is not None and math.hypot(v_forward, v_left) > 0.1:
            back_angle = math.atan2(-v_left, v_forward) + math.pi
        else:
            back_angle = math.pi
        lateral, heading = self.detector.detect(frame, back_angle)
        self.road_visible = lateral is not None
        v_right = -v_left
        if not self.road_visible:
            if integrate:
                self.forward_integral += self.speed_ki * (0.0 - v_forward) * dt
                self.forward_integral = clamp(self.forward_integral, self.max_horizontal_accel)
            if self.detector.nearest_road is not None and height:
                # Road visible but not under the drone: aim for the nearest road pixel
                metres_per_px = 2.0 * height * math.tan(self.down_fov / 2.0) / frame.shape[1]
                dx, dy = self.detector.nearest_road
                self.last_road_point = (-dy * metres_per_px, dx * metres_per_px)
            self.update_return_command(v_forward, v_right, dt)
            return

        # Lateral: road offset in metres (positive = road to the right)
        half_width_m = (height or 0.0) * math.tan(self.down_fov / 2.0)
        offset = lateral * half_width_m
        self.last_road_point = (0.0, offset)
        self.return_cmd = None
        if integrate:
            self.lateral_integral += self.lateral_ki * offset * dt
            self.lateral_integral = clamp(self.lateral_integral, self.max_horizontal_accel)
        force_right = self.lateral_kp * offset + self.lateral_integral - self.lateral_kd * v_right

        # Forward: slow down in turns and when far off the road centre
        slowdown = max(abs(heading) / 0.6, abs(offset) / 0.15)
        target_speed = self.cruise_speed * max(0.2, 1.0 - slowdown)
        speed_error = target_speed - v_forward
        if integrate:
            self.forward_integral += self.speed_ki * speed_error * dt
            self.forward_integral = clamp(self.forward_integral, self.max_horizontal_accel)
        force_forward = self.speed_kp * speed_error + self.forward_integral

        # Road turning right (positive heading) -> turn clockwise (negative yaw rate)
        yaw = -clamp(self.yaw_kp * heading, self.max_yaw_rate)
        self.follow_cmd = (clamp(force_forward, self.max_horizontal_accel),
                           -clamp(force_right, self.max_horizontal_accel),
                           yaw)

    def update_return_command(self, v_forward, v_right, dt):
        """Steer back to the nearest/last road point seen. With no road in view
        the point is moved with the drone's measured velocity (the drone
        doesn't yaw while the road is lost)."""
        if self.last_road_point is None:
            self.return_cmd = None
            return
        forward, right = self.last_road_point
        forward -= v_forward * dt
        right -= v_right * dt
        self.last_road_point = (forward, right)
        force_forward = (self.lateral_kp * forward - self.lateral_kd * v_forward
                         + self.forward_integral)
        force_right = self.lateral_kp * right - self.lateral_kd * v_right + self.lateral_integral
        self.return_cmd = (clamp(force_forward, self.max_horizontal_accel),
                           -clamp(force_right, self.max_horizontal_accel))

    # ---------------------------------------------------------------- control

    def altitude_force(self, dt):
        if self.height() is None:
            return 0.0
        error = self.target_altitude - self.height()
        return clamp(self.altitude_pid.update(error, dt), self.max_vertical_accel)

    def send_score(self, location, prediction):
        msg = String()
        msg.data = f'{self.team_id},{self.team_password},{location},{prediction}'
        self.score_pub.publish(msg)
        self.get_logger().info(f'Sent to /score_tracker: {msg.data}')

    def set_state(self, state):
        self.state = state
        self.state_start_time = self.sim_time
        self.get_logger().info(f'State -> {state}')

    def elapsed(self):
        return self.sim_time - self.state_start_time

    def publish_cmd(self, forward=0.0, left=0.0, up=0.0, yaw=0.0):
        """forward/left/up are accelerations [m/s^2]; they're sent as forces [N]."""
        # Wind feedforward (a force) on top of the PID forces
        if self.wind_feedforward and self.flying():
            ff_forward, ff_left = self.wind.feedforward(self.ground_velocity)
            self.wind_ff = (self.wind_ff_gain * ff_forward, self.wind_ff_gain * ff_left)
        else:
            self.wind_ff = (0.0, 0.0)
        max_total = self.mass * self.max_total_horizontal_accel
        forward = clamp(self.mass * forward + self.wind_ff[0], max_total)
        left = clamp(self.mass * left + self.wind_ff[1], max_total)
        up = self.mass * up
        self.last_cmd = (forward, left, up)

        twist = Twist()
        twist.linear.x = forward
        twist.linear.y = left
        twist.linear.z = up
        twist.angular.z = yaw
        self.cmd_pub.publish(twist)

    def publish_hold(self, dt):
        self.publish_cmd(self.hold_cmd[0], self.hold_cmd[1], self.altitude_force(dt))

    def control_loop(self):
        if self.sim_time is None:
            return
        dt = 0.0 if self.last_control_time is None else self.sim_time - self.last_control_time
        self.last_control_time = self.sim_time

        if self.state == 'WAIT_CLOCK':
            # Give publishers/subscribers time to register before sending anything
            self.set_state('REGISTER_DELAY')

        elif self.state == 'REGISTER_DELAY':
            if self.elapsed() >= 1.0 and self.altitude is not None:
                self.send_score(0, 'NA')  # start the timer
                self.altitude_pid.reset()
                self.set_state('CLIMB')

        elif self.state == 'CLIMB':
            self.publish_hold(dt)
            if abs(self.target_altitude - self.height()) < 0.05 and self.elapsed() > 1.0:
                self.set_state('FOLLOW')

        elif self.state == 'FOLLOW':
            if self.road_visible:
                self.target_altitude = self.cruise_altitude
                forward, left, yaw = self.follow_cmd
                self.publish_cmd(forward, left, self.altitude_force(dt), yaw)
            else:
                # Road lost: go back to where it was last seen (or hold position if
                # it was never seen) and climb slowly to see more of the ground
                self.target_altitude = min(self.max_search_altitude,
                                           self.target_altitude + 0.2 * dt)
                if self.return_cmd is not None:
                    self.publish_cmd(self.return_cmd[0], self.return_cmd[1],
                                     self.altitude_force(dt))
                else:
                    self.publish_hold(dt)
                self.get_logger().warn('Road lost, returning to last seen road and climbing',
                                       throttle_duration_sec=2.0)

            if self.elapsed() >= self.follow_duration:
                self.set_state('HOVER')

        elif self.state == 'HOVER':
            self.publish_hold(dt)
            if self.elapsed() >= 1.0:
                self.send_score(-1, 'NA')  # stop the timer
                self.set_state('DONE')

        elif self.state == 'DONE':
            # Keep holding position until shut down
            self.publish_hold(dt)

    # ---------------------------------------------------------------- debug

    def wind_text(self):
        wind = self.wind.wind_velocity(self.ground_velocity)
        if wind is None:
            return 'Wind: --'
        return (f'Wind {WindEstimator.speed(wind):.2f} m/s (fwd {wind[0]:+.2f} left {wind[1]:+.2f})'
                f'  ff {self.wind_ff[0]:+.1f} {self.wind_ff[1]:+.1f} N')

    def show_debug_window(self, frame):
        self.detector.draw_debug(frame)
        frame = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        height, source = self.height_reading()
        alt = f'{height:.2f} ({source})' if height is not None else '--'
        lat = self.detector.lateral_error
        head = self.detector.heading_error
        v = self.velocity
        vel = (f'Ground vel fwd {v.forward:+.2f} left {v.left:+.2f} m/s'
               if v is not None and v.valid else 'Ground vel: --')
        if self.road_visible:
            forward, left, yaw = self.follow_cmd
        else:
            forward, left, yaw = (*(self.return_cmd or self.hold_cmd), 0.0)
        lines = [f'State: {self.state}',
                 f'Alt: {alt} m (target {self.target_altitude:.2f})',
                 f'Lateral: {lat:+.2f}' if lat is not None else 'Lateral: NO ROAD',
                 f'Heading: {math.degrees(head):+.0f} deg' if head is not None else 'Heading: --',
                 vel,
                 self.wind_text(),
                 f'a fwd {forward:+.2f}  left {left:+.2f} m/s^2  yaw {yaw:+.2f}']
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2)
        cv2.imshow('Down Camera', frame)
        cv2.waitKey(1)


def main(args=None):
    # Handle Ctrl+C ourselves so the context is still valid to send a final command
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = DroneRoadFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Zero force command: the plugin keeps compensating gravity, so the drone hovers
        node.publish_cmd()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
