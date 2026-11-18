#!/usr/bin/env python3
"""Drone time trial controller.

Starts the score tracker timer, takes off, flies forward, hovers, then stops
the timer. The drone never lands again: once flying, every ground contact is
a -5 pt penalty.

Open loop: a fixed sequence of force commands on /cmd_vel
(linear.x/y/z = forward/left/up force in N). The drone plugin compensates
gravity and applies drag (1 N per m/s), so each "go" phase is followed by an
equal and opposite "brake" phase to stop the drone.

Topics used: /clock (sub), /cmd_vel, /score_tracker (pub).
"""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String


class DroneTimeTrial(Node):

    # (state name, duration [sim s], force (forward, left, up) [N])
    # With 1 kg and 1 N/(m/s) drag: CLIMB + BRAKE_CLIMB rises ~1.3 m,
    # FORWARD + BRAKE_FORWARD travels ~1.8 m.
    FLIGHT_PLAN = [
        ('CLIMB', 1.2, (0.0, 0.0, 2.0)),
        ('BRAKE_CLIMB', 0.5, (0.0, 0.0, -2.0)),
        ('STEADY', 1.0, (0.0, 0.0, 0.0)),
        ('FORWARD', 1.5, (2.0, 0.0, 0.0)),
        ('BRAKE_FORWARD', 0.55, (-2.0, 0.0, 0.0)),
        ('HOVER', 2.0, (0.0, 0.0, 0.0)),
    ]

    def __init__(self):
        super().__init__('drone_time_trial')

        # Team ID and password: max 8 characters each, no spaces or commas
        self.declare_parameter('team_id', 'TeamFizz')
        self.declare_parameter('team_password', 'pass1234')

        self.team_id = self.get_parameter('team_id').value
        self.team_password = self.get_parameter('team_password').value

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.score_pub = self.create_publisher(String, '/score_tracker', 1)
        self.create_subscription(Clock, '/clock', self.clock_callback, 1)

        self.sim_time = None
        self.state = 'WAIT_CLOCK'
        self.state_start_time = None
        self.plan_index = -1

        self.create_timer(0.05, self.control_loop)  # 20 Hz
        self.get_logger().info('Drone time trial started, waiting for /clock...')

    def clock_callback(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec / 1e9

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

    def publish_force(self, forward, left, up):
        twist = Twist()
        twist.linear.x = forward
        twist.linear.y = left
        twist.linear.z = up
        self.cmd_pub.publish(twist)

    def start_segment(self, index):
        self.plan_index = index
        self.set_state(self.FLIGHT_PLAN[index][0])

    def control_loop(self):
        if self.sim_time is None:
            return

        if self.state == 'WAIT_CLOCK':
            # Give publishers/subscribers time to register before sending anything
            self.set_state('REGISTER_DELAY')

        elif self.state == 'REGISTER_DELAY':
            if self.elapsed() >= 1.0:
                self.send_score(0, 'NA')  # start the timer
                self.start_segment(0)

        elif 0 <= self.plan_index < len(self.FLIGHT_PLAN):
            _, duration, force = self.FLIGHT_PLAN[self.plan_index]
            self.publish_force(*force)

            if self.elapsed() >= duration:
                if self.plan_index + 1 < len(self.FLIGHT_PLAN):
                    self.start_segment(self.plan_index + 1)
                else:
                    self.plan_index = len(self.FLIGHT_PLAN)
                    self.send_score(-1, 'NA')  # stop the timer
                    self.set_state('DONE')

        elif self.state == 'DONE':
            # Zero force: the plugin compensates gravity, so the drone hovers
            self.publish_force(0.0, 0.0, 0.0)


def main(args=None):
    # Handle Ctrl+C ourselves so the context is still valid to send a final command
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = DroneTimeTrial()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_force(0.0, 0.0, 0.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
