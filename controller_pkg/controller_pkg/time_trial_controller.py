#!/usr/bin/env python3
"""Time trial controller.

Starts the score tracker timer, follows the paved road with a PID
controller on the front camera image, stops, then stops the timer.

Only uses the allowed simulation topics:
  - subscribes to /clock and /camera/image_raw
  - publishes to /cmd_vel and /score_tracker
"""
import math

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import String

from controller_pkg.lane_follower import PID, LaneFollower


class TimeTrialController(Node):

    # State machine states
    WAIT_CLOCK = 'WAIT_CLOCK'
    REGISTER_DELAY = 'REGISTER_DELAY'
    DRIVE = 'DRIVE'
    STOPPING = 'STOPPING'
    DONE = 'DONE'

    def __init__(self):
        super().__init__('time_trial_controller')

        # Team ID and password: max 8 characters each, no spaces or commas
        self.declare_parameter('team_id', 'TeamFizz')
        self.declare_parameter('team_password', 'pass1234')
        self.declare_parameter('linear_speed', 0.4)     # m/s on a straight
        self.declare_parameter('drive_duration', 30.0)  # sim seconds of road following
        self.declare_parameter('kp', 2.0)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.15)
        self.declare_parameter('max_angular', 1.5)      # rad/s
        self.declare_parameter('show_camera', True)

        self.team_id = self.get_parameter('team_id').value
        self.team_password = self.get_parameter('team_password').value
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.drive_duration = float(self.get_parameter('drive_duration').value)
        self.max_angular = float(self.get_parameter('max_angular').value)
        self.show_camera = bool(self.get_parameter('show_camera').value)

        self.pid = PID(float(self.get_parameter('kp').value),
                       float(self.get_parameter('ki').value),
                       float(self.get_parameter('kd').value))
        self.lane_follower = LaneFollower()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.score_pub = self.create_publisher(String, '/score_tracker', 1)
        self.create_subscription(Clock, '/clock', self.clock_callback, 1)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 1)

        self.bridge = CvBridge()
        if self.show_camera:
            cv2.namedWindow('Robot Camera', cv2.WINDOW_NORMAL)

        self.sim_time = None
        self.state = self.WAIT_CLOCK
        self.state_start_time = None

        # Latest steering command computed from the camera
        self.error = None
        self.last_error = 0.0
        self.last_image_time = None
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0

        # 10 Hz control loop (wall time, decisions are made on sim time)
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Time trial controller started, waiting for /clock...')

    def clock_callback(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec / 1e9

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.update_steering(frame)

        if self.show_camera:
            self.lane_follower.draw_debug(frame)

            # Debug overlay: current state, sim time and steering
            sim_time_text = f'{self.sim_time:.1f}s' if self.sim_time is not None else '--'
            error_text = f'{self.error:+.2f}' if self.error is not None else 'NO ROAD'
            lines = [f'State: {self.state}',
                     f'Sim time: {sim_time_text}',
                     f'Error: {error_text}',
                     f'v: {self.cmd_linear:.2f}  w: {self.cmd_angular:+.2f}']
            for i, text in enumerate(lines):
                cv2.putText(frame, text, (10, 30 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Robot Camera', frame)
            cv2.waitKey(1)

    def update_steering(self, frame):
        """Run road detection + PID on a new frame and store the command."""
        self.error = self.lane_follower.compute_error(frame)

        now = self.sim_time
        dt = 0.0 if (now is None or self.last_image_time is None) else now - self.last_image_time
        self.last_image_time = now

        if self.error is None:
            # Road lost: stop and turn on the spot towards where it was last seen
            self.pid.reset()
            self.cmd_linear = 0.0
            self.cmd_angular = -math.copysign(0.5, self.last_error)
            return

        self.last_error = self.error
        # Positive error = road is to the right -> turn clockwise (negative z)
        angular = -self.pid.update(self.error, dt)
        self.cmd_angular = max(-self.max_angular, min(self.max_angular, angular))
        # Slow down in turns
        self.cmd_linear = self.linear_speed * max(0.3, 1.0 - abs(self.error))

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

    def publish_velocity(self, linear, angular=0.0):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

    def control_loop(self):
        if self.sim_time is None:
            return

        if self.state == self.WAIT_CLOCK:
            # Give publishers/subscribers time to register before sending anything
            self.set_state(self.REGISTER_DELAY)

        elif self.state == self.REGISTER_DELAY:
            if self.elapsed() >= 1.0:
                self.send_score(0, 'NA')  # start the timer
                self.pid.reset()
                self.set_state(self.DRIVE)

        elif self.state == self.DRIVE:
            if self.elapsed() < self.drive_duration:
                self.publish_velocity(self.cmd_linear, self.cmd_angular)
            else:
                self.publish_velocity(0.0)
                self.set_state(self.STOPPING)

        elif self.state == self.STOPPING:
            # Keep commanding zero velocity briefly so the robot settles
            self.publish_velocity(0.0)
            if self.elapsed() >= 1.0:
                self.send_score(-1, 'NA')  # stop the timer
                self.set_state(self.DONE)

        elif self.state == self.DONE:
            pass


def main(args=None):
    # Handle Ctrl+C ourselves so the context is still valid to send a final stop command
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = TimeTrialController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_velocity(0.0)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
