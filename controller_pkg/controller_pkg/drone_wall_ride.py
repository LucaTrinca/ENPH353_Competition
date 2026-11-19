#!/usr/bin/env python3
"""Drone wall ride: follow the arena walls from the spawn to the far corners.

Route (the wall is always on the drone's right):
  1. take off to the ride altitude drifting gently away from the south wall
     (only 0.25 m behind the spawn), turn to face +x, and once the right
     sensor sees the south wall hold the distance to it
  2. ride the south (long) wall to the east corner
  3. turn left 90 deg and ride the east (short) wall to the north-east corner
  4. turn left again and ride the north wall west to the tunnel at the foot of
     the mountain, low (0.28 m) and on the tunnel's centre line, until the
     front range sensor says it's right there, then push gently for 2 s:
     touching the tunnel earns the +5 bonus (no respawns before it). The
     touch point is known exactly (the tunnel's east face), so x and y are
     reset there. Clue 7 faces this approach, so the front camera reads it
     again, close up.
  5. up and over to clue 8: back off into open air, climb gently (0.2 m/s;
     optical flow over a 0.4 s baseline keeps working high up) to 2.3 m (altimeter; the
     plateau is 1.82 m), fly to the middle of the plateau facing west - the
     downward range sensor confirms it (the ground jumps up to ~0.5 m below)
     and keeps at least 0.3 m of clearance - then creep west, steering by
     clue 8's direction in the front camera, until it's big enough to read,
     and hold there
  6. done: hover; the timer is stopped once clue_finder has submitted

Clue 6 faces west, 0.64 m in from the south wall (y = -1.71, x random in
3.0 - 3.41), so it can't be seen from the wall line. It is found with
landmarks, not the dead-reckoned x (which drifts 30 - 60% over the lake):
  - when clue_finder has read clue 5 (location 5) from the left camera and
    the drone has gone clue6_clear_distance further (past the trees that
    stand on clue 6's line near clue 5), it steps out onto clue 6's line:
    wall distance 1.04 m, still measured by the right range sensor. (If clue
    5 is missed, it steps out when the dead-reckoned x passes
    clue6_fallback_x.)
  - it flies east along the line slowly until the front camera sees clue 6
    big enough to read (clue_finder's /drone/board_bearing), holds there so
    it can be read, then steps back to the wall and carries on. It gives up
    after clue6_timeout s. (Flying past it would go over it: the board's top
    is 0.20 m up, the drone rides at 0.35 m.)

Sensors:
  - /right_depth  - a 32 x 4 strip of depth pixels looking right. Each column
    is a different bearing, so a line fit gives both the distance to the wall
    and the wall's angle relative to the drone (used to correct the yaw).
  - /front_depth  - an 8 x 4 strip looking ahead, finds the corner wall
  - /altimeter    - ride height (above the start), flat except the lake
  - /imu + down camera optical flow - velocity, and the distance travelled
    along each wall (the front wall is only trusted after most of the wall's
    length, so trees in the way don't look like a corner)

The cameras and clue boards are handled by a separate process, clue_finder
(started by the same launch file), so camera work can never slow this
control loop down. This node tells it what it's doing on /drone/state and
/drone/status. At the final corner it waits for /drone/clues_flushed (the
last boards in view have been read and submitted) before stopping the timer.

Control works in accelerations (m/s^2), sent as forces (x drone_mass) on /cmd_vel.
The world's wind force is proportional to mass (WindEffects), so it pushes a
20 kg drone as hard as a light one - up to ~1.8 m/s^2 here. A disturbance
observer (IMU acceleration minus the commanded acceleration, low-passed)
measures that push and the controller cancels it.

Topics used: /clock, /imu, /altimeter, /down_camera/image_raw, /down_depth,
             /right_depth, /front_depth, /drone/clues_flushed (sub),
             /cmd_vel, /score_tracker, /drone/pose_estimate, /drone/state,
             /drone/status (pub).
"""
import math
import warnings

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from ros_gz_interfaces.msg import Altimeter
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Empty, String

from controller_pkg.course_map import START
from controller_pkg.imu_utils import level_acceleration
from controller_pkg.lane_follower import PID
from controller_pkg.optical_flow import GroundVelocityEstimator

# Sensor mounting (drone.urdf.xacro / drone_gazebo.xacro)
DOWN_CAMERA_HEIGHT_OFFSET = 0.01
RIGHT_RANGE_Y = -0.06     # m, right range sensor offset from the centre (body y)
RIGHT_RANGE_FOV = 1.2     # rad
FRONT_RANGE_X = 0.16      # m, front range sensor offset from the centre (body x)

# Walls ridden, in order: (name, wall direction = drone heading [rad],
# expected length [m] from where the leg starts to the corner stop point)
LEGS = [
    ('south wall', 0.0, 11.1),
    ('east wall', math.pi / 2.0, 4.8),
    ('north wall', math.pi, 1.9),       # to the tunnel (only with tunnel_and_clue8)
]
# States in which the drone is high above the ground (ground clearance floor applies)
HIGH_STATES = ('CLIMB', 'CROSS', 'CLUE8', 'CLUE8_HOLD', 'DONE')


def clamp(value, limit):
    return max(-limit, min(limit, value))


def wrap(angle):
    return math.remainder(angle, 2.0 * math.pi)


def fit_wall(depth, fov, max_range):
    """Fit a straight wall to a right-looking depth strip.

    Returns (distance from the sensor, angle) or None, where angle = drone
    heading - wall direction (negative = the drone is yawed toward the wall,
    so the wall gets closer going forward)."""
    rows = depth[1:-1] if depth.shape[0] > 2 else depth
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # columns with no valid pixel
        d = np.nanmedian(np.where(np.isfinite(rows), rows, np.nan), axis=0)
    width = d.size
    f = (width / 2.0) / math.tan(fov / 2.0)
    u = np.arange(width) + 0.5 - width / 2.0
    ok = np.isfinite(d) & (d > 0.1) & (d < max_range)
    if ok.sum() < 8:
        return None
    # Sensor frame: x = forward along the drone, y = depth (away to the right).
    # Image right = toward the drone's back.
    x = -d[ok] * u[ok] / f
    y = d[ok]
    # Least squares, then refit without the outliers (trees, cars)
    inliers = np.ones(x.size, bool)
    for _ in range(2):
        if inliers.sum() < 8:
            return None
        m, c = np.polyfit(x[inliers], y[inliers], 1)
        residual = np.abs(y - (m * x + c)) / math.sqrt(1.0 + m * m)
        inliers = residual < 0.03
    if inliers.sum() < 8 or np.ptp(x[inliers]) < 0.15:
        return None
    rms = float(np.sqrt(np.mean(residual[inliers] ** 2)))
    if rms > 0.015:
        return None
    # y = m x + c: the wall gets closer going forward when m < 0
    angle = math.atan(m)
    return c * math.cos(angle), angle


class DroneWallRide(Node):

    def __init__(self):
        super().__init__('drone_wall_ride')

        # Team ID and password: max 8 characters each, no spaces or commas
        self.declare_parameter('team_id', 'TeamFizz')
        self.declare_parameter('team_password', 'pass1234')
        self.declare_parameter('drone_mass', 20.0)          # kg, must match the URDF
        self.declare_parameter('down_camera_fov_deg', 134.0)
        self.declare_parameter('ride_altitude', 0.35)       # m above the start (walls are 0.5 m tall)
        self.declare_parameter('wall_distance', 0.4)        # m, drone centre to the wall
        self.declare_parameter('corner_distance', 0.4)      # m, drone centre to the wall ahead at a corner
        self.declare_parameter('cruise_speed', 0.35)        # m/s along the walls
        self.declare_parameter('corner_gate', 0.6)          # fraction of a wall's length before the wall ahead counts
        self.declare_parameter('max_wall_range', 1.5)       # m, right range readings beyond this are ignored
        self.declare_parameter('wall_search_speed', 0.15)   # m/s sideways toward the wall when it's lost
        # Gains and limits are accelerations (m/s^2 per unit of error)
        self.declare_parameter('wall_kp', 3.0)              # per m
        self.declare_parameter('wall_kd', 3.0)              # per m/s
        self.declare_parameter('speed_kp', 1.5)             # per m/s
        self.declare_parameter('approach_gain', 0.8)        # m/s of speed per m from the corner stop point
        self.declare_parameter('position_kp', 3.0)          # hover / takeoff hold, per m
        self.declare_parameter('position_kd', 4.0)          # per m/s
        self.declare_parameter('max_horizontal_accel', 3.0)  # controller output, before the wind cancel
        self.declare_parameter('disturbance_tau', 0.3)      # s, low-pass of the measured push (0 = off)
        self.declare_parameter('max_disturbance', 2.0)      # m/s^2 (the strongest gusts are ~1.8)
        self.declare_parameter('altitude_kp', 3.0)
        self.declare_parameter('altitude_ki', 0.5)
        self.declare_parameter('altitude_kd', 3.0)
        self.declare_parameter('max_vertical_accel', 3.0)
        self.declare_parameter('max_climb_rate', 0.3)       # m/s
        self.declare_parameter('yaw_kp', 2.0)               # rad/s per rad
        self.declare_parameter('max_yaw_rate', 0.6)         # rad/s
        self.declare_parameter('wall_yaw_gain', 0.05)       # yaw correction per wall fit (30 Hz)
        self.declare_parameter('flow_gain', 0.3)
        self.declare_parameter('flow_max_yaw_rate', 0.2)
        self.declare_parameter('flow_max_vertical_speed', 0.15)
        self.declare_parameter('flow_crop', 0.35)           # keeps the wall beside the drone out of view
        self.declare_parameter('flow_min_texture', 6.0)
        self.declare_parameter('sensor_timeout', 0.3)       # s
        self.declare_parameter('contact_decay_tau', 1.0)    # s, wind estimate fades while touching a wall
        # Tunnel and clue 8 (see the module docstring)
        self.declare_parameter('tunnel_and_clue8', True)
        self.declare_parameter('tunnel_wall_distance', 0.45)  # m from the north wall = the tunnel's line (y 2.3)
        self.declare_parameter('tunnel_altitude', 0.28)     # m, low enough to hit the arch's east face
        self.declare_parameter('tunnel_speed', 0.25)        # m/s, 0.08 once the tunnel is within 0.6 m
        self.declare_parameter('tunnel_touch_time', 2.0)    # s of gentle push after reaching it
        self.declare_parameter('tunnel_touch_x', 3.73)      # m, drone centre x when touching its east face
        self.declare_parameter('tunnel_timeout', 40.0)      # s on the north wall before giving up
        self.declare_parameter('cruise_altitude', 2.3)      # m above the start, crossing to the plateau
        self.declare_parameter('high_climb_rate', 0.2)      # m/s, gentle so the optical flow keeps working
        # Aim a little west of the plateau's middle (2.0): the dead-reckoned cross
        # tends to fall ~0.8 m short (the down camera sees ground at very
        # different depths here), and the approach to clue 8 is visual anyway
        self.declare_parameter('plateau_x', 1.7)            # m, plateau (1.82 m high) spans x 1.0 - 2.5
        self.declare_parameter('plateau_y', 1.2)
        self.declare_parameter('cross_speed', 0.3)          # m/s
        self.declare_parameter('agl_floor', 0.3)            # m, minimum ground clearance when high up
        self.declare_parameter('plateau_range', 0.7)        # m, ground closer than this below = over the plateau
        self.declare_parameter('clue8_speed', 0.15)         # m/s west across the plateau
        self.declare_parameter('clue8_stop_area', 0.03)     # board fraction of the front image to stop at
        self.declare_parameter('clue8_dwell', 3.0)          # s held in front of it
        self.declare_parameter('clue8_timeout', 30.0)       # s before giving up
        self.declare_parameter('test_climb', False)         # testing: climb right after takeoff
        # Clue 6 visit (see the module docstring)
        self.declare_parameter('visit_clue6', True)
        self.declare_parameter('clue6_wall_distance', 1.04)  # m from the south wall = the board's line (y -1.71)
        self.declare_parameter('clue6_speed', 0.25)         # m/s along the line
        self.declare_parameter('clue6_stop_area', 0.03)     # board fraction of the front image to stop at
        self.declare_parameter('clue6_dwell', 3.0)          # s held in front of the board
        self.declare_parameter('clue6_timeout', 25.0)       # s on the line before giving up
        self.declare_parameter('clue6_clear_distance', 0.6)  # m further after reading clue 5 (past its trees)
        self.declare_parameter('clue6_fallback_x', 2.5)     # m (dead-reckoned), step out here if clue 5 is missed
        self.declare_parameter('clue_wait', 5.0)            # s to wait for clue_finder before stopping the timer

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.team_id = p('team_id')
        self.team_password = p('team_password')
        self.mass = float(p('drone_mass'))
        self.down_fov = math.radians(float(p('down_camera_fov_deg')))
        self.ride_altitude = float(p('ride_altitude'))
        self.wall_distance = float(p('wall_distance'))
        self.corner_distance = float(p('corner_distance'))
        self.cruise_speed = float(p('cruise_speed'))
        self.corner_gate = float(p('corner_gate'))
        self.max_wall_range = float(p('max_wall_range'))
        self.wall_search_speed = float(p('wall_search_speed'))
        self.wall_kp = float(p('wall_kp'))
        self.wall_kd = float(p('wall_kd'))
        self.speed_kp = float(p('speed_kp'))
        self.approach_gain = float(p('approach_gain'))
        self.position_kp = float(p('position_kp'))
        self.position_kd = float(p('position_kd'))
        self.max_horizontal_accel = float(p('max_horizontal_accel'))
        self.disturbance_tau = float(p('disturbance_tau'))
        self.max_disturbance = float(p('max_disturbance'))
        self.max_vertical_accel = float(p('max_vertical_accel'))
        self.max_climb_rate = float(p('max_climb_rate'))
        self.yaw_kp = float(p('yaw_kp'))
        self.max_yaw_rate = float(p('max_yaw_rate'))
        self.wall_yaw_gain = float(p('wall_yaw_gain'))
        self.flow_gain = float(p('flow_gain'))
        self.flow_max_yaw_rate = float(p('flow_max_yaw_rate'))
        self.flow_max_vertical_speed = float(p('flow_max_vertical_speed'))
        self.flow_crop = float(p('flow_crop'))
        self.flow_min_texture = float(p('flow_min_texture'))
        self.sensor_timeout = float(p('sensor_timeout'))
        self.contact_decay_tau = float(p('contact_decay_tau'))
        self.tunnel_and_clue8 = bool(p('tunnel_and_clue8'))
        self.tunnel_wall_distance = float(p('tunnel_wall_distance'))
        self.tunnel_altitude = float(p('tunnel_altitude'))
        self.tunnel_speed = float(p('tunnel_speed'))
        self.tunnel_touch_time = float(p('tunnel_touch_time'))
        self.tunnel_touch_x = float(p('tunnel_touch_x'))
        self.tunnel_timeout = float(p('tunnel_timeout'))
        self.cruise_altitude = float(p('cruise_altitude'))
        self.high_climb_rate = float(p('high_climb_rate'))
        self.plateau = (float(p('plateau_x')), float(p('plateau_y')))
        self.cross_speed = float(p('cross_speed'))
        self.agl_floor = float(p('agl_floor'))
        self.plateau_range = float(p('plateau_range'))
        self.clue8_speed = float(p('clue8_speed'))
        self.clue8_stop_area = float(p('clue8_stop_area'))
        self.clue8_dwell = float(p('clue8_dwell'))
        self.clue8_timeout = float(p('clue8_timeout'))
        self.test_climb = bool(p('test_climb'))
        self.visit_clue6 = bool(p('visit_clue6'))
        self.clue6_wall_distance = float(p('clue6_wall_distance'))
        self.clue6_speed = float(p('clue6_speed'))
        self.clue6_stop_area = float(p('clue6_stop_area'))
        self.clue6_dwell = float(p('clue6_dwell'))
        self.clue6_timeout = float(p('clue6_timeout'))
        self.clue6_clear_distance = float(p('clue6_clear_distance'))
        self.clue6_fallback_x = float(p('clue6_fallback_x'))
        self.clue_wait = float(p('clue_wait'))

        # Small integral limit: gravity is already cancelled by the plugin, the
        # integral only trims small biases (a big one winds up if the drone is
        # ever held down, then overshoots above the 0.5 m walls)
        self.altitude_pid = PID(float(p('altitude_kp')), float(p('altitude_ki')),
                                float(p('altitude_kd')), integral_limit=0.4)
        self.velocity_estimator = None
        self.bridge = CvBridge()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.score_pub = self.create_publisher(String, '/score_tracker', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/drone/pose_estimate', 1)
        self.state_pub = self.create_publisher(String, '/drone/state', 10)
        self.status_pub = self.create_publisher(String, '/drone/status', 1)
        self.create_subscription(Clock, '/clock', self.clock_callback, 1)
        self.create_subscription(Imu, '/imu', self.imu_callback, 50)
        self.create_subscription(Altimeter, '/altimeter', self.altimeter_callback, 1)
        self.create_subscription(Image, '/down_depth', self.down_range_callback, 1)
        self.create_subscription(Image, '/right_depth', self.right_range_callback, 1)
        self.create_subscription(Image, '/front_depth', self.front_range_callback, 1)
        self.create_subscription(Image, '/down_camera/image_raw', self.down_image_callback, 1)
        self.create_subscription(Empty, '/drone/clues_flushed', self.clues_flushed_callback, 1)
        self.create_subscription(PointStamped, '/drone/board_bearing', self.board_bearing_callback, 10)
        self.create_subscription(String, '/drone/board_identified', self.board_identified_callback, 10)

        self.sim_time = None
        self.state = 'WAIT_CLOCK'
        self.state_start_time = None
        self.last_control_time = None

        # Height
        self.altitude = None
        self.vertical_speed = 0.0
        self.ground_level = None
        self.range_height = None
        self.range_time = None
        self.altitude_setpoint = 0.0

        # Walls
        self.wall = None            # (distance from the drone centre, angle), latest good fit
        self.wall_time = None
        self.front = None           # distance from the drone centre to the wall ahead
        self.front_time = None

        # Dead-reckoned state in the course frame (from the known start pose)
        self.x, self.y, self.yaw = START
        self.vx = self.vy = 0.0
        self.yaw_rate = 0.0
        self.last_imu_time = None
        self.imu_gaps = 0
        # Wind / disturbance observer (level body frame, m/s^2)
        self.command_accel = (0.0, 0.0)
        self.disturbance = (0.0, 0.0)

        self.stop_pending_since = None   # sim time DONE was reached, timer not stopped yet
        # Clue 6 visit: pending -> line (step out, fly east) -> hold -> done
        self.clue6_phase = 'pending' if self.visit_clue6 else 'done'
        self.clue6_since = None
        self.clue5_read_x = None         # x estimate when clue 5 was read
        self.front_board_area = 0.0      # latest front-camera board size (fraction of the image)
        self.front_board_bearing = 0.0   # rad, positive = right of the image centre
        self.front_board_time = None
        # Tunnel and clue 8
        self.altitude_target = self.ride_altitude
        self.touch_since = None
        self.cross_start = None
        self.cross_since = None
        self.plateau_seen = False
        self.plateau_since = None       # over the plateau (continuously) since, during the clue 8 approach
        self.off_edge_since = None
        self.legs = len(LEGS) if self.tunnel_and_clue8 else 2
        self.last_flow = None            # (forward, left) m/s, last optical-flow measurement used

        # Route progress
        self.leg = 0
        self.leg_start = (self.x, self.y)
        self.hold_point = (self.x, self.y)
        self.target_yaw = self.yaw
        self.settled_time = None

        self.create_timer(0.05, self.control_loop)  # 20 Hz
        self.create_timer(0.2, self.publish_status)
        self.get_logger().info('Drone wall ride started, waiting for /clock...')

    # ---------------------------------------------------------------- sensors

    def clock_callback(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec / 1e9

    def fresh(self, stamp):
        return (stamp is not None and self.sim_time is not None
                and self.sim_time - stamp <= self.sensor_timeout)

    def altimeter_callback(self, msg):
        if self.ground_level is None:
            self.ground_level = msg.vertical_position
        self.altitude = msg.vertical_position - self.ground_level
        self.vertical_speed = msg.vertical_velocity

    def down_range_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        valid = depth[np.isfinite(depth)]
        if valid.size:
            self.range_height = float(np.median(valid))
            self.range_time = self.sim_time

    def height_above_ground(self):
        return self.range_height if self.fresh(self.range_time) else self.altitude

    def right_range_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        fit = fit_wall(depth, RIGHT_RANGE_FOV, self.max_wall_range)
        if fit is None:
            # Closer than the sensor's 0.1 m near clip reads as -inf
            if np.mean(np.isneginf(depth)) > 0.25:
                self.wall = (0.1 - RIGHT_RANGE_Y, 0.0)
                self.wall_time = self.sim_time
            return
        distance, angle = fit
        self.wall = (distance - RIGHT_RANGE_Y * math.cos(angle), angle)
        self.wall_time = self.sim_time
        # The wall's direction is known, so its angle gives the drone's heading
        if self.state == 'RIDE':
            measured_yaw = LEGS[self.leg][1] + angle
            error = wrap(measured_yaw - self.yaw)
            if abs(error) < 0.3:
                self.yaw = wrap(self.yaw + self.wall_yaw_gain * error)

    def front_range_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        # Top rows look slightly up (the bottom row can hit the ground far ahead),
        # central columns only
        w = depth.shape[1]
        patch = depth[: max(1, depth.shape[0] // 2), w // 4: w - w // 4]
        valid = patch[np.isfinite(patch) & (patch > 0.1)]
        if np.mean(np.isneginf(patch)) > 0.25:
            # Closer than the 0.1 m near clip (reads as -inf), not "nothing ahead"
            self.front = 0.1 + FRONT_RANGE_X
            self.front_time = self.sim_time
        elif valid.size:
            self.front = float(np.median(valid)) + FRONT_RANGE_X
            self.front_time = self.sim_time

    def in_contact(self, margin=0.22):
        """A range sensor says the drone's frame (0.15 m half width) is at a wall."""
        return ((self.fresh(self.wall_time) and self.wall[0] < margin)
                or (self.fresh(self.front_time) and self.front < margin + 0.05))

    def flying(self):
        h = self.height_above_ground()
        return h is not None and h > 0.05 and self.state not in ('WAIT_CLOCK', 'REGISTER_DELAY')

    def imu_callback(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        dt = 0.0 if self.last_imu_time is None else stamp - self.last_imu_time
        self.last_imu_time = stamp
        if dt <= 0.0 or dt > 0.5:
            return
        if dt > 0.02:
            self.imu_gaps += 1  # missed samples (100 Hz IMU); the step still integrates dt
        self.yaw_rate = msg.angular_velocity.z
        ax, ay = level_acceleration(msg)  # gravity leak from tilt removed
        if not self.flying():
            self.vx = self.vy = 0.0
            self.disturbance = (0.0, 0.0)
            return
        # Disturbance observer: what accelerated the drone that we didn't command
        # (wind, mostly), in the level body frame
        if self.altitude is None or self.altitude < 0.15:
            # Still near the ground (ground contact is not wind)
            self.disturbance = tuple(0.9 * d for d in self.disturbance)
        elif self.in_contact():
            # Touching a wall: the contact force is not wind, so don't learn from
            # it (that would push harder into the wall). Fade the estimate out
            # rather than freezing it - frozen at a saturated value it pinned the
            # drone against the wall
            fade = max(0.0, 1.0 - dt / self.contact_decay_tau)
            self.disturbance = tuple(fade * d for d in self.disturbance)
        elif self.disturbance_tau > 0.0:
            a = min(1.0, dt / self.disturbance_tau)
            self.disturbance = tuple(
                clamp(d + a * ((m - c) - d), self.max_disturbance)
                for d, m, c in zip(self.disturbance, (ax, ay), self.command_accel))
        self.yaw = wrap(self.yaw + self.yaw_rate * dt)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.vx += (c * ax - s * ay) * dt
        self.vy += (s * ax + c * ay) * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def down_image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.velocity_estimator is None:
            # Low: consecutive frames, centre crop (keeps the wall beside the drone out)
            self.velocity_estimator = GroundVelocityEstimator(
                self.down_fov, frame.shape[1], smoothing=0.3,
                crop_fraction=self.flow_crop, min_texture=self.flow_min_texture)
            # High (over 1 m): the ground moves well under a pixel per frame, so use
            # the whole image at full resolution and a 0.4 s baseline
            self.high_velocity_estimator = GroundVelocityEstimator(
                self.down_fov, frame.shape[1], smoothing=0.3, crop_fraction=1.0,
                min_texture=self.flow_min_texture, downscale=1.0, baseline=0.4)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        h = self.height_above_ground()
        if not self.flying() or h is None:
            self.velocity_estimator.reset()
            self.high_velocity_estimator.reset()
            return
        if h > 1.0:
            self.velocity_estimator.reset()
            estimator = self.high_velocity_estimator
        else:
            self.high_velocity_estimator.reset()
            estimator = self.velocity_estimator
        velocity = estimator.update(frame, stamp, h + DOWN_CAMERA_HEIGHT_OFFSET)
        # Climbing zooms the image. High up, over the 0.4 s baseline a gentle climb
        # (0.2 m/s) zooms it only a few %, so the flow is still used then.
        max_vertical = 0.3 if h > 1.0 else self.flow_max_vertical_speed
        if (velocity is None or abs(self.yaw_rate) > self.flow_max_yaw_rate
                or abs(self.vertical_speed) > max_vertical):
            return
        forward, left = velocity
        self.last_flow = (forward, left)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.vx += self.flow_gain * ((c * forward - s * left) - self.vx)
        self.vy += self.flow_gain * ((s * forward + c * left) - self.vy)

    # ---------------------------------------------------------------- control

    def send_score(self, location, prediction):
        msg = String()
        msg.data = f'{self.team_id},{self.team_password},{location},{prediction}'
        self.score_pub.publish(msg)
        self.get_logger().info(f'Sent to /score_tracker: {msg.data}')

    def stop_timer(self):
        if self.stop_pending_since is not None:
            self.stop_pending_since = None
            self.send_score(-1, 'NA')  # stop the timer

    def board_bearing_callback(self, msg):
        # Front camera: how big the board ahead looks (stops the clue 6 approach)
        if msg.header.frame_id == 'front':
            self.front_board_area = msg.point.y
            self.front_board_bearing = msg.point.x
            self.front_board_time = self.sim_time

    def board_identified_callback(self, msg):
        # "camera,location,TYPE,VALUE" from clue_finder
        parts = msg.data.split(',')
        if (len(parts) >= 2 and parts[1] == '5' and self.state == 'RIDE' and self.leg == 0
                and self.clue5_read_x is None):
            self.clue5_read_x = self.x
            self.get_logger().info('Clue 5 read - clue 6 next')

    def clues_flushed_callback(self, _msg):
        # clue_finder has read and submitted the boards still in view
        if self.state == 'DONE':
            self.stop_timer()

    def publish_status(self):
        # State for clue_finder (it only looks for boards while riding), and a
        # status line for its camera window
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)
        wall = f'wall {self.wall[0]:.2f} m' if self.fresh(self.wall_time) else 'wall -'
        front = f'front {self.front:.2f} m' if self.fresh(self.front_time) else 'front -'
        leg = LEGS[self.leg][0] if self.state == 'RIDE' else ''
        msg = String()
        msg.data = (f'{self.state} {leg}  est x {self.x:+.2f} y {self.y:+.2f} '
                    f'yaw {math.degrees(self.yaw):+.0f}  {wall}  {front}')
        self.status_pub.publish(msg)

    def set_state(self, state):
        self.state = state
        msg = String()
        msg.data = state
        self.state_pub.publish(msg)
        self.state_start_time = self.sim_time
        self.settled_time = None
        self.get_logger().info(f'State -> {state} (estimate x {self.x:+.2f} y {self.y:+.2f} '
                               f'yaw {math.degrees(self.yaw):+.0f} deg)')

    def body_velocity(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * self.vx + s * self.vy, -s * self.vx + c * self.vy

    def publish(self, a_forward, a_left, yaw_target, dt):
        """Body-frame horizontal acceleration, altitude hold and yaw P -> /cmd_vel."""
        norm = math.hypot(a_forward, a_left)
        if norm > self.max_horizontal_accel:
            a_forward *= self.max_horizontal_accel / norm
            a_left *= self.max_horizontal_accel / norm
        # Cancel the measured push (wind)
        a_forward -= self.disturbance[0]
        a_left -= self.disturbance[1]
        self.command_accel = (a_forward, a_left)
        rate = self.high_climb_rate if self.state == 'CLIMB' else self.max_climb_rate
        self.altitude_setpoint += clamp(self.altitude_target - self.altitude_setpoint, rate * dt)
        # High up, never get closer to the ground than agl_floor (downward range)
        if (self.state in HIGH_STATES and self.fresh(self.range_time)
                and self.range_height < self.agl_floor):
            self.altitude_setpoint = max(self.altitude_setpoint,
                                         self.altitude + self.agl_floor - self.range_height)
        a_up = clamp(self.altitude_pid.update(self.altitude_setpoint - self.altitude, dt),
                     self.max_vertical_accel)
        twist = Twist()
        twist.linear.x = self.mass * a_forward
        twist.linear.y = self.mass * a_left
        twist.linear.z = self.mass * a_up
        twist.angular.z = clamp(self.yaw_kp * wrap(yaw_target - self.yaw), self.max_yaw_rate)
        self.cmd_pub.publish(twist)

    def hold(self, point, yaw_target, dt):
        """Hold a dead-reckoned course-frame point."""
        ex, ey = point[0] - self.x, point[1] - self.y
        ax = self.position_kp * ex - self.position_kd * self.vx
        ay = self.position_kp * ey - self.position_kd * self.vy
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.publish(c * ax + s * ay, -s * ax + c * ay, yaw_target, dt)

    def settled(self, condition, duration=1.0):
        """True once `condition` has held for `duration` seconds."""
        if not condition:
            self.settled_time = None
            return False
        if self.settled_time is None:
            self.settled_time = self.sim_time
        return self.sim_time - self.settled_time >= duration

    def ride(self, dt):
        """Ride the wall on the right; returns True at the corner."""
        name, heading, length = LEGS[self.leg]
        v_forward, v_left = self.body_velocity()

        # Clue 6 visit (south wall only): after clue 5, step out onto the board's
        # line and fly east until the front camera sees it big enough; hold; back
        visiting = False
        wall_target = self.wall_distance
        if self.leg == 0 and self.state == 'RIDE' and self.clue6_phase != 'done':
            # Trees stand on clue 6's line just past clue 5 (x -2 .. 0.2), so go a
            # little further before stepping out (a short distance: drift is small)
            past_clue5 = (self.clue5_read_x is not None
                          and self.x - self.clue5_read_x > self.clue6_clear_distance)
            if self.clue6_phase == 'pending' and (past_clue5 or self.x > self.clue6_fallback_x):
                self.clue6_phase = 'line'
                self.clue6_since = self.sim_time
                self.get_logger().info('Clue 6: stepping out onto its line '
                                       + ('(past clue 5)' if past_clue5
                                          else f'(clue 5 not read - estimate x {self.x:+.2f})'))
            if self.clue6_phase in ('line', 'hold'):
                visiting = True
                wall_target = self.clue6_wall_distance

        if self.leg == 2:
            wall_target = self.tunnel_wall_distance

        # Sideways: hold the distance to the wall (it grows when moving left)
        if self.fresh(self.wall_time):
            a_left = -self.wall_kp * (self.wall[0] - wall_target) - self.wall_kd * v_left
        else:
            # Lost the wall (too far away, or a tree in the way): the wall is
            # always on the right, so drift right slowly until it's seen again
            a_left = -self.wall_kd * (v_left + self.wall_search_speed)

        # Along the wall: cruise, then stop at the wall ahead once most of this
        # wall's length has been covered
        travelled = math.hypot(self.x - self.leg_start[0], self.y - self.leg_start[1])
        speed = self.cruise_speed
        corner_seen = travelled > self.corner_gate * length and self.fresh(self.front_time)
        error = 0.0
        if corner_seen:
            error = self.front - self.corner_distance
            speed = clamp(self.approach_gain * error, self.cruise_speed)
        if self.leg == 2 and self.state == 'RIDE':
            # To the tunnel: slow down close to it, then push gently against it
            corner_seen = False
            near = self.fresh(self.front_time) and self.front < 0.6
            touching = self.fresh(self.front_time) and self.front <= 0.27   # within the 0.1 m near clip
            if touching and self.touch_since is None:
                self.touch_since = self.sim_time
                self.get_logger().info('Tunnel: reached it, pushing gently')
            if self.touch_since is not None:
                a_forward = 0.3                      # ~6 N forward, keeps the frame against the arch
            else:
                speed = 0.08 if near else self.tunnel_speed
                a_forward = self.speed_kp * (speed - v_forward)
            self.publish(a_forward, a_left, heading, dt)
            if self.touch_since is not None and self.sim_time - self.touch_since >= self.tunnel_touch_time:
                return True
            if self.sim_time - self.state_start_time > self.tunnel_timeout:
                self.get_logger().warn('Tunnel: not reached, carrying on to clue 8 anyway')
                return True
            return False
        if visiting:
            corner_seen = False
            elapsed = self.sim_time - self.clue6_since
            if self.clue6_phase == 'line':
                # Slowly east along the line (once on it) until the board is big enough
                on_line = self.fresh(self.wall_time) and abs(self.wall[0] - wall_target) < 0.15
                speed = self.clue6_speed if on_line else 0.0
                a_forward = self.speed_kp * (speed - v_forward)
                seen = (self.front_board_time is not None and self.sim_time - self.front_board_time < 1.0
                        and self.front_board_area >= self.clue6_stop_area)
                if seen:
                    self.clue6_phase = 'hold'
                    self.clue6_since = self.sim_time
                    self.get_logger().info(f'Clue 6: in front of it ({self.front_board_area * 100:.1f}% '
                                           f'of the front image), holding {self.clue6_dwell:.0f} s')
                elif elapsed > self.clue6_timeout:
                    self.clue6_phase = 'done'
                    self.get_logger().warn('Clue 6: not found, back to the wall')
            else:
                a_forward = -self.position_kd * v_forward  # hold still
                if elapsed >= self.clue6_dwell:
                    self.clue6_phase = 'done'
                    self.get_logger().info('Clue 6: done, back to the wall')
        elif corner_seen and error < 0.4:
            # Close to the corner: hold the distance to the wall ahead as stiffly
            # as the wall beside (PD on the front range)
            a_forward = self.wall_kp * error - self.wall_kd * v_forward
        else:
            a_forward = self.speed_kp * (speed - v_forward)
        self.publish(a_forward, a_left, heading, dt)
        wall = (f'{self.wall[0]:.2f} m {math.degrees(self.wall[1]):+.0f} deg'
                if self.fresh(self.wall_time) else 'lost')
        front = f'{self.front:.2f} m' if self.fresh(self.front_time) else 'none'
        self.get_logger().info(
            f'{name}: travelled {travelled:.2f} m, v fwd {v_forward:+.2f} left {v_left:+.2f}, '
            f'wall {wall}, front {front}, wind push {self.disturbance[0]:+.2f} '
            f'{self.disturbance[1]:+.2f} m/s^2, IMU gaps {self.imu_gaps}', throttle_duration_sec=1.0)

        if self.sim_time - self.state_start_time > length / self.cruise_speed + 30.0:
            self.get_logger().warn(f'{name}: no corner found after {travelled:.1f} m', once=True)
        return self.settled(corner_seen and abs(error) < 0.08 and abs(v_forward) < 0.1, 0.5)

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

        elapsed = 0.0 if self.state_start_time is None else self.sim_time - self.state_start_time
        if self.state == 'WAIT_CLOCK':
            self.set_state('REGISTER_DELAY')
        elif self.state == 'REGISTER_DELAY':
            if elapsed >= 1.0 and self.altitude is not None:
                self.send_score(0, 'NA')  # start the timer
                self.altitude_pid.reset()
                self.altitude_setpoint = self.altitude
                self.hold_point = (self.x, self.y)
                self.set_state('TAKEOFF')
        elif self.state == 'TAKEOFF':
            # The south wall is right behind the spawn: drift gently forward,
            # away from it, while climbing (a gust can push the drone onto it)
            v_forward, v_left = self.body_velocity()
            self.publish(self.speed_kp * (0.1 - v_forward), -self.position_kd * v_left, START[2], dt)
            # Hover 2 s at the top still facing north: clue 1 faces the spawn, so
            # the front camera gets a few frames of it before the turn
            if self.settled(abs(self.altitude - self.altitude_target) < 0.03, 2.0) or elapsed > 12.0:
                self.hold_point = (self.x, self.y)
                if self.test_climb:
                    self.altitude_target = self.cruise_altitude
                    self.set_state('CLIMB')
                else:
                    self.set_state('TURN')
        elif self.state == 'TURN':
            if self.leg == 2:
                self.altitude_target = self.tunnel_altitude
            # Turn on the spot to face along the next wall. Once the right
            # sensor sees that wall, hold the distance to it (the spawn is only
            # 0.25 m from the south wall, and a gust can push the drone onto it)
            heading = LEGS[self.leg][1]
            if self.fresh(self.wall_time) and abs(wrap(heading - self.yaw)) < 0.4:
                v_forward, v_left = self.body_velocity()
                a_left = -self.wall_kp * (self.wall[0] - self.wall_distance) - self.wall_kd * v_left
                self.publish(-self.position_kd * v_forward, a_left, heading, dt)
            else:
                self.hold(self.hold_point, heading, dt)
            if self.settled(abs(wrap(heading - self.yaw)) < 0.05, 0.5) or elapsed > 10.0:
                self.leg_start = (self.x, self.y)
                self.set_state('RIDE')
                self.get_logger().info(f'Riding the {LEGS[self.leg][0]}')
        elif self.state == 'RIDE':
            if self.ride(dt):
                self.hold_point = (self.x, self.y)
                self.get_logger().info(f'Corner at the end of the {LEGS[self.leg][0]} '
                                       f'(travelled {math.hypot(self.x - self.leg_start[0], self.y - self.leg_start[1]):.2f} m)')
                if self.leg == 2:
                    # Touched the tunnel's east face: a known point, so reset x and y
                    wall = self.wall[0] if self.fresh(self.wall_time) else self.tunnel_wall_distance
                    self.x, self.y = self.tunnel_touch_x, 2.75 - wall
                    self.vx = self.vy = 0.0
                    self.get_logger().info(f'Touched the tunnel (bonus +5) - position reset to '
                                           f'x {self.x:+.2f} y {self.y:+.2f}')
                    # Back off into open air before climbing: 0.5 m east (off the
                    # tunnel) and 0.5 m south (off the north wall) - pressed against a
                    # wall, its friction can stall the climb
                    self.hold_point = (self.x + 0.5, self.y - 0.5)
                    self.set_state('BACK_OFF')
                elif self.leg + 1 < self.legs:
                    self.leg += 1
                    self.set_state('TURN')
                else:
                    # clue_finder reads and submits the boards still in view, then
                    # says so on /drone/clues_flushed and the timer is stopped
                    self.stop_pending_since = self.sim_time
                    self.set_state('DONE')
        if self.state in HIGH_STATES or self.state == 'BACK_OFF':
            flow = 'none' if self.last_flow is None else f'{self.last_flow[0]:+.2f} {self.last_flow[1]:+.2f}'
            rng = f'{self.range_height:.2f}' if self.fresh(self.range_time) else '-'
            self.get_logger().info(
                f'{self.state}: est x {self.x:+.2f} y {self.y:+.2f} v {self.vx:+.2f} {self.vy:+.2f} '
                f'yaw {math.degrees(self.yaw):+.0f}, alt {self.altitude:.2f} vz {self.vertical_speed:+.2f}, '
                f'range {rng}, flow fwd/left {flow}, wind {self.disturbance[0]:+.2f} '
                f'{self.disturbance[1]:+.2f}', throttle_duration_sec=1.0)
        if self.state == 'BACK_OFF':
            self.hold(self.hold_point, math.pi, dt)
            if self.settled(math.hypot(self.hold_point[0] - self.x, self.hold_point[1] - self.y) < 0.08,
                            0.5) or elapsed > 6.0:
                self.hold_point = (self.x, self.y)
                self.altitude_target = self.cruise_altitude
                self.set_state('CLIMB')
        elif self.state == 'CLIMB':
            # Straight up in open air, in front of the tunnel
            self.hold(self.hold_point, math.pi, dt)
            if self.settled(abs(self.altitude - self.cruise_altitude) < 0.05, 0.5) or elapsed > 15.0:
                self.cross_start = (self.x, self.y)
                self.plateau_seen = False
                self.set_state('CROSS')
        elif self.state == 'CROSS':
            # A point moving at cross_speed from the climb point to the plateau
            sx, sy = self.cross_start
            tx, ty = self.plateau
            length = math.hypot(tx - sx, ty - sy)
            k = min(1.0, self.cross_speed * elapsed / max(length, 1e-6))
            self.hold((sx + k * (tx - sx), sy + k * (ty - sy)), math.pi, dt)
            if (not self.plateau_seen and self.fresh(self.range_time)
                    and self.range_height < self.plateau_range):
                self.plateau_seen = True
                self.get_logger().info(f'Over the plateau (ground {self.range_height:.2f} m below)')
            if k >= 1.0 and math.hypot(tx - self.x, ty - self.y) < 0.15:
                if not self.plateau_seen:
                    self.get_logger().warn('Plateau not confirmed by the range sensor - looking for clue 8 anyway')
                self.plateau_since = self.off_edge_since = None
                self.set_state('CLUE8')
        elif self.state == 'CLUE8':
            # West across the plateau, centring clue 8 in the front camera
            v_forward, v_left = self.body_velocity()
            seen = self.front_board_time is not None and self.sim_time - self.front_board_time < 1.0
            v_left_target = clamp(-0.8 * self.front_board_bearing, 0.15) if seen else 0.0
            a_forward = self.speed_kp * (self.clue8_speed - v_forward)
            a_left = self.speed_kp * (v_left_target - v_left)
            self.publish(a_forward, a_left, math.pi, dt)
            # The plateau's west edge (clue 8 stands near it): over the plateau for
            # 3 s, then the ground drops away for 0.5 s. (At an edge the range
            # sensor flickers between the two heights, so neither is instant.)
            if self.fresh(self.range_time):
                if self.range_height < self.plateau_range:
                    if self.plateau_since is None:
                        self.plateau_since = self.sim_time
                    self.off_edge_since = None
                elif self.range_height > self.plateau_range + 0.4:
                    if self.plateau_since is not None and self.sim_time - self.plateau_since > 3.0:
                        if self.off_edge_since is None:
                            self.off_edge_since = self.sim_time
                    else:
                        self.plateau_since = None
            off_edge = self.off_edge_since is not None and self.sim_time - self.off_edge_since > 0.5
            if seen and self.front_board_area >= self.clue8_stop_area:
                self.get_logger().info(f'Clue 8: in front of it ({self.front_board_area * 100:.1f}% of '
                                       f'the front image), holding {self.clue8_dwell:.0f} s')
                self.set_state('CLUE8_HOLD')
            elif off_edge:
                self.get_logger().warn('Clue 8: reached the plateau edge without it - stopping')
                self.set_state('CLUE8_HOLD')
            elif elapsed > self.clue8_timeout:
                self.get_logger().warn('Clue 8: not found')
                self.set_state('CLUE8_HOLD')
        elif self.state == 'CLUE8_HOLD':
            v_forward, v_left = self.body_velocity()
            self.publish(-self.position_kd * v_forward, -self.position_kd * v_left, math.pi, dt)
            if elapsed >= self.clue8_dwell:
                self.hold_point = (self.x, self.y)
                # clue_finder reads and submits the boards still in view, then
                # says so on /drone/clues_flushed and the timer is stopped
                self.stop_pending_since = self.sim_time
                self.set_state('DONE')
        elif self.state == 'DONE':
            # Don't wait for clue_finder for ever (it may not be running)
            if (self.stop_pending_since is not None
                    and self.sim_time - self.stop_pending_since > self.clue_wait):
                self.get_logger().warn('No word from clue_finder, stopping the timer anyway')
                self.stop_timer()
            if self.tunnel_and_clue8:
                self.hold(self.hold_point, math.pi, dt)   # hover where clue 8 was read
            else:
                self.ride(dt)   # keep riding the last wall's corner: holds both wall distances


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = DroneWallRide()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
