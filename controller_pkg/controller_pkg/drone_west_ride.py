#!/usr/bin/env python3
"""Second drone: clues 1 - 4 in the west half of the arena.

Route (course frame: x east, y north):

  1. clue 1 at (-5.81, -1.64), faces south: read during a short hover after
     takeoff, from a fixed point south of it (the spawn is right south of it)
  2. clue 2 at (-5.16, 1.11 .. 1.6), faces south: ride north with the west wall
     on the LEFT at west_wall_distance (0.5 m), stop when the front camera
     first sees it, turn to face it, hold. Not from straight in front: the fir
     at (-5, 0.9) stands right south of it, so the drone looks at it ~25 deg
     from the side (west)
  3. clue 3 at (-4.0, 1.67), faces west: on up the west wall to the north-west
     corner (front range to the north wall), turn east there, along the north
     wall (y 2.25, north of the fir at (-5, 1.8)) and down to 0.65 m in front
     of clue 3: the front camera reads it head-on
  4. clue 4 at (-0.83, 0.0 .. 0.63), faces north: diagonally to the south-west
     of the road loop, east along a lane just south of it (y 0.05 - the truck
     drives the loop) to 0.5 m from the plateau's west face, turn to -45 deg
     (the right camera then looks south-west) and slide north until the board
     is in the middle of the right camera's image: the line 45 deg off the
     board's facing. Straight north of the board stands an oak (-0.92, 0.92)
     that hides it from there
  5. land where it is (out of the first drone's way)

The truck (0.08 m tall) drives around the loop and the pedestrian crosses at
y -0.5: the drone flies over the truck (0.35 m) and never near the crossing.

Position. The west half is flat green paint and asphalt: too little texture
for the optical flow (it often measures nothing, and along a straight road
edge only the motion across it), so the route is laid out for the range
sensors, and the dead reckoning only bridges short gaps between their fixes:
  - facing north on the west wall line: x from the west wall (left range), y
    from the north wall (front range)
  - facing east near the north wall: y from the north wall (left range)
  - facing roughly east (within 50 deg) south of the loop or on the north wall
    line: x from the plateau's west face (front range, a plane at x 0.14 at the
    sensor's height, clear down both lanes)
Turns happen only where a fix follows straight away: in the north-west corner,
and at the end of the lane south of the loop.

The wall line is 0.5 m from the west wall on purpose: a row of fir trees
stands at x = -5.0 (y -2, -0.9, 0, 0.9, 1.8). Their crowns are 0.51 m tall and
0.17 m in radius (only a ~5 cm collision, so the drone flies through them), and
they confuse the optical flow and the range sensors. It never touches the
competition timer (the first drone starts and stops it), and its clue_finder
only submits clues 1 - 4.

States: WAIT_CLOCK -> REGISTER_DELAY (wait for drone 1 to leave) -> TAKEOFF ->
CLUE1 -> RIDE (west wall, north) -> FACE (turn to clue 2) -> CLUE_HOLD ->
TO_CORNER -> TURN_EAST -> NORTH_WALL -> TO_CLUE3 -> CLUE3 -> TO_LOOP -> BOTTOM
-> TURN4 -> SLIDE -> CLUE4_HOLD -> DONE (land). NORTH_WALL / TO_* / BOTTOM are
point-to-point legs (turn on the spot to the leg's heading, then fly to its end
point). The stuck watchdog, the wind cancel, optical flow and altitude hold are
drone_wall_ride's; every leg also has a timeout, so a stuck leg moves on.

Runs in a namespace (drone2): every topic is relative except /clock and
/score_tracker. Needs the drone spawned with prefix:=drone2/ left_range:=true
(drone2.launch.py does it).

Topics (relative): cmd_vel, imu, altimeter, down_depth, left_depth,
front_depth, down_camera/image_raw, drone/board_bearing (from its clue_finder),
drone/state, drone/status, drone/pose_estimate, drone/clues_flushed.
"""
import collections
import math

import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String

from controller_pkg.drone_wall_ride import DroneWallRide, fit_wall, wrap

LEFT_RANGE_Y = 0.06      # left range sensor, 0.06 m left of the centre (drone.urdf.xacro)
LEFT_RANGE_FOV = 1.2
NORTH = math.pi / 2.0
EAST = 0.0
WEST_WALL_X = -6.0       # arena walls (collision planes, fast_empty.sdf)
NORTH_WALL_Y = 2.75
NORTH_LANE_Y = 2.0       # north of this, facing east, both lanes to the walls are clear
# Point-to-point legs: state -> (end point, heading)
LEGS = {
    'NORTH_WALL': ('north_lane_end', EAST),   # east along the north wall
    'TO_CLUE3': ('clue3_view', EAST),         # down to 0.65 m in front of clue 3
    'TO_LOOP': ('loop_head', EAST),           # south-west of the loop
    'BOTTOM': ('clue4_line', EAST),           # east along the lane south of the loop
}
ROUTE_STATES = ('TO_CORNER', 'TURN_EAST') + tuple(LEGS) + ('CLUE3', 'TURN4', 'SLIDE', 'CLUE4_HOLD')


class RangeTrack:
    """Recent readings of one range to one known surface. A straight line
    through them over time (the drone moves steadily between fixes) gives the
    distance now and its rate of change; readings that don't fit one (a tree
    passing through the beam, the beam dipping to the ground while the drone
    tilts) give nothing."""

    def __init__(self, length, tolerance):
        self.history = collections.deque(maxlen=length)
        self.tolerance = tolerance
        self.key = None

    def clear(self):
        self.history.clear()
        self.key = None

    def add(self, key, stamp, reading):
        """Returns (distance now, rate in m/s) or None."""
        if key != self.key:
            self.clear()
            self.key = key
        self.history.append((stamp, reading))
        if len(self.history) < self.history.maxlen:
            return None
        t = np.array([h[0] for h in self.history])
        r = np.array([h[1] for h in self.history])
        t = t - t[-1]
        if np.ptp(t) <= 0.0:
            return None
        rate, now = np.polyfit(t, r, 1)
        if np.max(np.abs(r - (rate * t + now))) > self.tolerance:
            return None
        return float(now), float(rate)


class DroneWestRide(DroneWallRide):

    def __init__(self):
        super().__init__()
        self.declare_parameter('start_x', -5.82)            # spawn (drone2.launch.py)
        self.declare_parameter('start_y', -1.85)
        self.declare_parameter('takeoff_hover', 0.5)        # s settled at ride height
        # Clue 1 (-5.81, -1.64) is read from a fixed point south of it, facing
        # north: a gust at liftoff can push the drone 0.6 m, out of its view
        self.declare_parameter('clue1_view_x', -5.78)
        self.declare_parameter('clue1_view_y', -2.3)
        self.declare_parameter('clue1_dwell', 2.5)          # s after first seeing it
        self.declare_parameter('clue1_timeout', 8.0)
        self.declare_parameter('west_wall_distance', 0.5)   # m, drone centre to the west wall (clear of the firs)
        self.declare_parameter('pull_out_distance', 0.8)    # m north over which it moves out to west_wall_distance
        self.declare_parameter('west_speed', 0.3)           # m/s north, no board in view
        self.declare_parameter('board_speed', 0.15)         # m/s north while a board is in view
        # Where to look for clue 2 (y from the front range to the north wall):
        # clue 2 stands at y 1.11 - 1.6 and the front camera sees 0.2 - 1.2 m ahead
        self.declare_parameter('clue2_search_y', -0.6)      # boards seen before this are clue 1
        self.declare_parameter('clue2_give_up_y', 1.9)      # past every clue 2 position
        self.declare_parameter('range_fix_gain', 0.1)       # per reading, pulling x / y to the wall ranges
        self.declare_parameter('range_velocity_gain', 0.15)  # per front reading (15 Hz), velocity from its rate
        self.declare_parameter('face_timeout', 6.0)         # s to turn toward a board
        # Take off only once the first drone has left the spawn box (a gust at
        # liftoff can push either drone 0.5 m): its state RIDE for this long
        self.declare_parameter('drone1_clear_time', 6.0)    # s
        self.declare_parameter('drone1_wait', 5.0)          # s without word from drone 1: go anyway
        self.declare_parameter('drone1_state_topic', '/drone/state')
        self.declare_parameter('clue2_dwell', 3.0)          # s in front of clue 2
        # Clues 3 and 4
        self.declare_parameter('clues_3_4', True)           # False: stop after clue 2
        self.declare_parameter('leg_speed', 0.4)            # m/s on the point-to-point legs
        self.declare_parameter('bottom_speed', 0.5)         # m/s along the lane south of the loop
        self.declare_parameter('corner_front', 0.5)         # m to the north wall in the north-west corner (y 2.25)
        self.declare_parameter('north_lane_end_x', -4.65)   # along the north wall to above clue 3's viewing point
        self.declare_parameter('north_lane_end_y', 2.25)
        self.declare_parameter('clue3_view_x', -4.65)       # 0.65 m west of clue 3 (-4.0, 1.67)
        self.declare_parameter('clue3_view_y', 1.67)
        self.declare_parameter('clue3_dwell', 2.5)          # s after first seeing it
        self.declare_parameter('clue3_timeout', 8.0)
        # South of the loop: the truck drives round the loop (its bottom side at
        # y 0.5), and under the down camera it drags the optical flow along
        self.declare_parameter('loop_head_x', -3.6)         # south-west of the loop (clear of the fir at (-4.1, 0.3))
        self.declare_parameter('loop_head_y', 0.05)
        self.declare_parameter('plateau_face_x', 0.14)      # plateau's west face at the range sensor's height (terrain mesh)
        self.declare_parameter('clue4_line_distance', 0.5)  # m from the plateau face: the slide line, x -0.36
        self.declare_parameter('clue4_line_y', 0.05)        # the lane south of the loop (0.45 m from the truck's)
        # While sliding the right camera looks at -135 deg: 45 deg off the board's
        # facing, so the line of sight passes south-east of the oak (-0.92, 0.92)
        # even with the board at its northernmost (y 0.63)
        self.declare_parameter('clue4_heading_deg', -45.0)
        self.declare_parameter('slide_speed', 0.12)         # m/s north
        self.declare_parameter('slide_max_y', 1.35)         # past the last clue 4 viewing point (0.63 + 0.47)
        self.declare_parameter('slide_gain', 0.8)           # m/s north per rad of the board right of the image centre
        self.declare_parameter('clue4_dwell', 3.0)          # s in front of clue 4
        self.declare_parameter('land_when_done', True)      # land at the end (out of drone 1's way)
        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.x, self.y, self.yaw = float(p('start_x')), float(p('start_y')), NORTH
        self.leg_start = self.hold_point = (self.x, self.y)
        self.takeoff_hover = float(p('takeoff_hover'))
        self.clue1_view = (float(p('clue1_view_x')), float(p('clue1_view_y')))
        self.clue1_dwell = float(p('clue1_dwell'))
        self.clue1_timeout = float(p('clue1_timeout'))
        self.seen_since = None
        self.last_bearing_time = None
        self.west_wall_distance = float(p('west_wall_distance'))
        self.pull_out_distance = float(p('pull_out_distance'))
        self.ride_start_wall = None
        self.face_yaw = NORTH
        self.west_speed = float(p('west_speed'))
        self.board_speed = float(p('board_speed'))
        self.clue2_search_y = float(p('clue2_search_y'))
        self.clue2_give_up_y = float(p('clue2_give_up_y'))
        self.range_fix_gain = float(p('range_fix_gain'))
        self.range_velocity_gain = float(p('range_velocity_gain'))
        self.face_timeout = float(p('face_timeout'))
        self.drone1_clear_time = float(p('drone1_clear_time'))
        self.drone1_wait = float(p('drone1_wait'))
        self.front_track = RangeTrack(6, 0.03)     # 0.4 s at 15 Hz
        self.left_track = RangeTrack(9, 0.03)      # 0.3 s at 30 Hz
        self.drone1_state = None
        self.drone1_ride_since = None
        self.create_subscription(String, p('drone1_state_topic'), self.drone1_state_callback, 10)
        self.create_timer(1.0, self.log_flow)
        self.clue2_dwell = float(p('clue2_dwell'))
        self.clues_3_4 = bool(p('clues_3_4'))
        self.leg_speed = float(p('leg_speed'))
        self.bottom_speed = float(p('bottom_speed'))
        self.corner_front = float(p('corner_front'))
        self.plateau_face_x = float(p('plateau_face_x'))
        self.points = {prefix: (float(p(prefix + '_x')), float(p(prefix + '_y')))
                       for prefix in ('north_lane_end', 'clue3_view', 'loop_head')}
        self.points['clue4_line'] = (self.plateau_face_x - float(p('clue4_line_distance')),
                                     float(p('clue4_line_y')))
        self.clue3_dwell = float(p('clue3_dwell'))
        self.clue3_timeout = float(p('clue3_timeout'))
        self.clue4_heading = math.radians(float(p('clue4_heading_deg')))
        self.slide_speed = float(p('slide_speed'))
        self.slide_max_y = float(p('slide_max_y'))
        self.slide_gain = float(p('slide_gain'))
        self.clue4_dwell = float(p('clue4_dwell'))
        self.land_when_done = bool(p('land_when_done'))
        self.route = collections.deque()
        self.goal = None            # (x, y, heading, speed, timeout) of the current leg
        self.goal_turned = False
        self.slide_x = None
        self.landed = False
        self.right_board_time = None
        self.right_board_bearing = 0.0
        self.right_board_area = 0.0
        self.watched_states = ('TAKEOFF', 'RIDE', 'TO_CORNER', 'SLIDE') + tuple(LEGS)
        self.wall_on_left = True
        self.fix_stats = collections.Counter()   # range fixes tried / used, logged each second
        self.create_subscription(Image, 'left_depth', self.left_range_callback, 1)
        self.get_logger().info('Drone 2 (west, clues 1 - 4) started, waiting for /clock...')

    # ---------------------------------------------------------------- sensors

    def right_range_callback(self, msg):
        # The right side looks into the open arena on this route: not used
        pass

    def left_range_callback(self, msg):
        """Wall on the left: mirror the strip and fit it like the right one."""
        depth = np.fliplr(self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1'))
        fit = fit_wall(depth, LEFT_RANGE_FOV, self.max_wall_range)
        if fit is None:
            if np.mean(np.isneginf(depth)) > 0.25:     # closer than the 0.1 m near clip
                self.wall = (0.1 + LEFT_RANGE_Y, 0.0)
                self.wall_time = self.sim_time
            return
        distance, angle = fit
        self.wall = (distance + LEFT_RANGE_Y * math.cos(angle), angle)
        self.wall_time = self.sim_time
        # Known walls: fixes for the dead reckoning (position, and velocity
        # from how fast the distance changes)
        if self.facing(NORTH) and (self.x < -4.8 or self.y > NORTH_LANE_Y):
            # West wall line, or the north-west corner: the west wall
            track = self.left_track.add('west', self.sim_time, self.wall[0])
            if track is not None:
                self.fix_position(x=WEST_WALL_X + track[0], vx=track[1], gain=0.5)
        elif self.facing(EAST) and self.y > 1.1:
            # The north wall. South of the north lane a fir can be in the strip
            # (fitted as a wall it is far off the estimate)
            track = self.left_track.add('north', self.sim_time, self.wall[0])
            if track is not None and (self.y > NORTH_LANE_Y or abs(NORTH_WALL_Y - track[0] - self.y) < 0.3):
                self.fix_position(y=NORTH_WALL_Y - track[0], vy=-track[1], gain=0.5)
        else:
            self.left_track.clear()
        # Mirrored, so the wall's angle gives the heading with the opposite sign
        if self.state == 'RIDE':
            error = wrap((NORTH - angle) - self.yaw)
            if abs(error) < 0.3:
                self.yaw = wrap(self.yaw + self.wall_yaw_gain * error)

    def front_range_callback(self, msg):
        super().front_range_callback(msg)
        # The front range is a narrow beam (0.3 rad) at the drone's height: it
        # passes over the 0.2 m boards and the truck
        steady_turn = abs(self.yaw_rate) < 0.2 and self.state not in ('WAIT_CLOCK', 'REGISTER_DELAY')
        if not (steady_turn and not self.landed and self.fresh(self.front_time) and 0.3 < self.front < 7.0):
            self.front_track.clear()
            return
        if self.facing(NORTH) and self.x < -4.4:
            axis = 'y'          # west of the firs it sees the north wall
        elif abs(wrap(self.yaw - EAST)) < 0.9:
            # The plateau's west face (a plane), whatever the exact heading
            axis = 'x'
        else:
            self.front_track.clear()
            return
        track = self.front_track.add(axis, self.sim_time, self.front)
        self.fix_stats[axis + ' tried'] += 1
        if track is None:
            return
        reading, rate = track
        if axis == 'y':
            self.fix_position(y=NORTH_WALL_Y - reading, vy=-rate)
            return
        c = math.cos(self.yaw)
        fix = self.plateau_face_x - reading * c
        # Clear lanes to the face: the north wall line, and south of the loop
        # (the oak at (-0.92, 0.92) is near the beam there, but only if the
        # estimate is 0.25 m off to the north). Elsewhere the loop's trees are
        # in the way: then only a fix close to the estimate counts
        bottom_lane = -0.3 < self.y < 0.9 and self.x > -4.0
        if self.y > NORTH_LANE_Y or bottom_lane or abs(fix - self.x) < 0.3:
            self.fix_position(x=fix, vx=-rate * c)
            self.fix_stats['x used'] += 1

    def board_bearing_callback(self, msg):
        super().board_bearing_callback(msg)
        if msg.header.frame_id == 'right':
            self.right_board_bearing = msg.point.x
            self.right_board_area = msg.point.y
            self.right_board_time = self.sim_time

    def drone1_state_callback(self, msg):
        if msg.data == 'RIDE' and self.drone1_ride_since is None and self.sim_time is not None:
            self.drone1_ride_since = self.sim_time
        self.drone1_state = msg.data

    def drone1_clear(self, waited):
        """The first drone has left the spawn box (or isn't flying at all)."""
        if self.drone1_state is None:
            return waited >= self.drone1_wait
        return (self.drone1_ride_since is not None
                and self.sim_time - self.drone1_ride_since >= self.drone1_clear_time)

    def log_flow(self):
        frames, measured, used, edge_only = self.flow_stats
        self.flow_stats = [0, 0, 0, 0]
        fixes = ' '.join(f'{k} {v}' for k, v in sorted(self.fix_stats.items()))
        self.fix_stats.clear()
        if self.state in ('WAIT_CLOCK', 'REGISTER_DELAY', None) or self.landed:
            return
        flow = 'none' if self.last_flow is None else f'{self.last_flow[0]:+.2f} {self.last_flow[1]:+.2f}'
        wall = f'{self.wall[0]:.2f}' if self.fresh(self.wall_time) else '-'
        front = f'{self.front:.2f}' if self.fresh(self.front_time) else '-'
        self.get_logger().info(
            f'{self.state}: est x {self.x:+.2f} y {self.y:+.2f} yaw {math.degrees(self.yaw):+.0f} '
            f'v {self.vx:+.2f} {self.vy:+.2f}, left {wall} front {front}, '
            f'flow frames {frames} measured {measured} used {used} (edge only {edge_only}), last flow {flow}, '
            f'alt {self.altitude:.2f} (set {self.altitude_setpoint:.2f}), '
            f'wind {self.disturbance[0]:+.2f} {self.disturbance[1]:+.2f}, front fixes: {fixes or "none"}')

    def facing(self, heading):
        return (self.state not in ('WAIT_CLOCK', 'REGISTER_DELAY') and not self.landed
                and abs(wrap(self.yaw - heading)) < 0.15 and abs(self.yaw_rate) < 0.2)

    def fix_position(self, x=None, y=None, vx=None, vy=None, gain=1.0):
        """Pull the estimate toward a range-based fix (position, and the
        velocity measured from the range's rate). `gain` scales both for
        faster sensors."""
        k = self.range_fix_gain * gain
        kv = self.range_velocity_gain * gain
        if x is not None:
            self.x += k * (x - self.x)
        if y is not None:
            self.y += k * (y - self.y)
        if vx is not None:
            self.vx += kv * (vx - self.vx)
        if vy is not None:
            self.vy += kv * (vy - self.vy)

    def high(self):
        # Never over the plateau: no ground clearance floor (it would stop the landing)
        return False

    def publish_status(self):
        msg = String()
        msg.data = self.state or ''
        self.state_pub.publish(msg)     # for clue_finder (a missed state change is resent)
        msg = String()
        msg.data = f'drone 2 {self.state}  est x {self.x:+.2f} y {self.y:+.2f} yaw {math.degrees(self.yaw):+.0f}'
        self.status_pub.publish(msg)

    # ---------------------------------------------------------------- control

    def control_loop(self):
        if self.sim_time is None:
            return
        dt = 0.0 if self.last_control_time is None else self.sim_time - self.last_control_time
        self.last_control_time = self.sim_time
        self.publish_pose()
        elapsed = 0.0 if self.state_start_time is None else self.sim_time - self.state_start_time
        v_forward, v_left = self.body_velocity()

        if self.state == 'WAIT_CLOCK':
            self.set_state('REGISTER_DELAY')
        elif self.state == 'REGISTER_DELAY':
            # (The first drone starts the timer - this one never touches it)
            if elapsed >= 1.0 and self.altitude is not None and self.drone1_clear(elapsed):
                self.get_logger().info('Taking off' + (' (drone 1 is clear of the spawn box)'
                                                       if self.drone1_state else ' (no drone 1)'))
                self.altitude_pid.reset()
                self.altitude_setpoint = self.altitude
                self.hold_point = (self.x, self.y)
                self.set_state('TAKEOFF')
        elif self.state == 'TAKEOFF':
            # Straight up holding the spawn point (the first drone is 0.6 m away),
            # then hover facing clue 1 (just north of the spawn) for the front camera
            self.hold(self.hold_point, NORTH, dt)
            if (self.settled(abs(self.altitude - self.altitude_target) < 0.03, self.takeoff_hover)
                    or elapsed > 12.0):
                self.face_yaw = NORTH
                self.seen_since = None
                self.set_state('CLUE1')
        elif self.state == 'CLUE1':
            # Go to the viewing point; once clue 1 is seen, turn to centre it
            if self.read_board(1, NORTH, self.clue1_view, self.clue1_dwell, self.clue1_timeout, elapsed, dt):
                self.start_ride()
        elif self.state == 'RIDE':
            self.ride_west(dt, v_forward, v_left)
        elif self.state == 'FACE':
            # Turn on the spot until clue 2 is in the middle of the front image
            seen = self.board_in_view()
            if seen:
                self.aim_at_board(NORTH)
            self.hold(self.hold_point, self.face_yaw, dt)
            centred = seen and abs(self.front_board_bearing) < 0.06
            if self.settled(centred, 0.5) or elapsed > self.face_timeout:
                self.get_logger().info(f'Clue 2: facing it ({self.front_board_area * 100:.1f}% of the front '
                                       f'image), holding {self.clue2_dwell:.0f} s')
                self.set_state('CLUE_HOLD')
        elif self.state == 'CLUE_HOLD':
            self.hold(self.hold_point, self.face_yaw, dt)
            if elapsed >= self.clue2_dwell:
                self.after_clue2()
        elif self.state == 'TO_CORNER':
            if self.to_corner(dt, v_forward, v_left) or elapsed > 25.0:
                self.next_step()
        elif self.state in ('TURN_EAST', 'TURN4'):
            heading = EAST if self.state == 'TURN_EAST' else self.clue4_heading
            self.hold(self.hold_point, heading, dt)
            if self.settled(abs(wrap(heading - self.yaw)) < 0.08, 0.3) or elapsed > 8.0:
                self.next_step()
        elif self.state in LEGS:
            arrived = self.fly_leg(dt)
            if arrived or elapsed > self.goal[4]:
                if not arrived:
                    self.get_logger().warn(f'{self.state}: not there after {elapsed:.0f} s '
                                           f'({math.hypot(self.goal[0] - self.x, self.goal[1] - self.y):.2f} m '
                                           f'to go) - going on')
                self.next_step()
        elif self.state == 'CLUE3':
            if self.read_board(3, EAST, self.hold_point, self.clue3_dwell, self.clue3_timeout, elapsed, dt):
                self.next_step()
        elif self.state == 'SLIDE':
            self.slide(dt, elapsed)
        elif self.state == 'CLUE4_HOLD':
            self.servo_clue4(dt)
            if elapsed >= self.clue4_dwell:
                self.finish()
        elif self.state == 'RECOVER':
            self.hold(self.recover_point, self.recover_yaw, dt)
            reached = math.hypot(self.recover_point[0] - self.x, self.recover_point[1] - self.y) < 0.1
            if self.settled(reached, 0.5) or elapsed > self.recover_time:
                self.resume()
        elif self.state == 'DONE':
            self.done(dt)
        self.watchdog()

    def done(self, dt):
        if not self.land_when_done:
            self.hold(self.hold_point, self.yaw, dt)
            return
        if self.landed:
            # Motors on, pushing down: a landed drone stays put in the wind
            twist = Twist()
            twist.linear.z = -2.0 * self.mass
            self.cmd_pub.publish(twist)
            return
        self.altitude_target = -0.1     # below the ground: it lands whatever the altimeter's offset
        self.hold(self.hold_point, self.yaw, dt)
        if self.settled(self.altitude < 0.03 and abs(self.vertical_speed) < 0.05, 1.0):
            self.landed = True
            self.get_logger().info(f'Landed (estimate x {self.x:+.2f} y {self.y:+.2f})')

    def publish_pose(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'course'
        pose.pose.position.x, pose.pose.position.y = self.x, self.y
        pose.pose.orientation.z, pose.pose.orientation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        self.pose_pub.publish(pose)

    def board_in_view(self, camera='front'):
        stamp = self.front_board_time if camera == 'front' else self.right_board_time
        return stamp is not None and self.sim_time - stamp < 1.0

    def aim_at_board(self, centre, limit=0.6):
        """Point at the board in the front camera, within `limit` of `centre`.
        Only on a new bearing (each camera frame): re-applying an old bearing
        every control tick kept turning the drone past the board - once all
        the way round to east."""
        if self.front_board_time == self.last_bearing_time:
            return
        self.last_bearing_time = self.front_board_time
        target = wrap(self.yaw - self.front_board_bearing)
        self.face_yaw = wrap(centre + max(-limit, min(limit, wrap(target - centre))))

    def read_board(self, clue, heading, point, dwell, timeout, elapsed, dt):
        """Hold a viewing point facing `heading`, turning to centre the board in
        the front camera once seen. True after `dwell` s with it in view, or at
        the timeout."""
        if self.board_in_view():
            self.aim_at_board(heading)
            if self.seen_since is None:
                self.seen_since = self.sim_time
                self.get_logger().info(f'Clue {clue}: in view ({self.front_board_area * 100:.1f}% of the '
                                       f'front image)')
        self.hold(point, self.face_yaw, dt)
        done = self.seen_since is not None and self.sim_time - self.seen_since >= dwell
        if not done and elapsed > timeout:
            self.get_logger().warn(f'Clue {clue}: not seen from the viewing point')
            return True
        return done

    def start_ride(self):
        self.leg_start = (self.x, self.y)
        self.ride_start_wall = self.wall[0] if self.fresh(self.wall_time) else None
        self.set_state('RIDE')
        self.get_logger().info('Riding the west wall north')

    def ride_west(self, dt, v_forward, v_left):
        """North along the west wall (on the left) until the front camera sees clue 2."""
        travelled = math.hypot(self.x - self.leg_start[0], self.y - self.leg_start[1])
        # Move out to the riding distance gradually while going north: straight
        # sideways from the spawn would head for the first drone
        target = self.west_wall_distance
        if self.ride_start_wall is not None:
            k = min(1.0, travelled / self.pull_out_distance)
            target = self.ride_start_wall + k * (self.west_wall_distance - self.ride_start_wall)
        board_seen = self.board_in_view()
        board_near = (self.board_seen_time is not None
                      and self.sim_time - self.board_seen_time < self.board_slow_time)
        on_line = self.fresh(self.wall_time) and abs(self.wall[0] - target) < 0.15
        speed = self.west_speed if on_line and not board_near else self.board_speed
        self.follow_west_wall(target, speed, dt, v_forward, v_left)

        wall = f'{self.wall[0]:.2f} m' if self.fresh(self.wall_time) else 'lost'
        self.get_logger().info(
            f'west wall: travelled {travelled:.2f} m, v fwd {v_forward:+.2f} (target {speed:.2f}) '
            f'left {v_left:+.2f}, wall {wall}, front board '
            f'{self.front_board_area * 100 if board_seen else 0:.1f}%', throttle_duration_sec=1.0)

        if board_seen and self.y > self.clue2_search_y:
            self.get_logger().info(f'Clue 2: in view ({self.front_board_area * 100:.1f}% of the front '
                                   f'image, travelled {travelled:.2f} m) - stopping to face it')
            self.hold_point = (self.x, self.y)
            self.face_yaw = self.yaw
            self.set_state('FACE')
        elif self.y > self.clue2_give_up_y:
            self.get_logger().warn(f'Clue 2: not found by y {self.y:+.2f} - going on')
            self.after_clue2()

    def follow_west_wall(self, target, speed, dt, v_forward, v_left):
        """Facing north at `target` from the west wall (on the left), `speed` north."""
        if self.fresh(self.wall_time):
            # Wall on the left: too far away -> move left (+)
            a_left = self.wall_kp * (self.wall[0] - target) - self.wall_kd * v_left
        else:
            a_left = -self.wall_kd * (v_left - self.wall_search_speed)   # drift left to find it
        a_forward = self.speed_kp * (speed - v_forward)
        self.publish(a_forward, a_left, NORTH, dt)

    def after_clue2(self):
        if self.clues_3_4:
            self.route = collections.deque(['TO_CORNER', 'TURN_EAST', 'NORTH_WALL', 'TO_CLUE3', 'CLUE3',
                                            'TO_LOOP', 'BOTTOM', 'TURN4', 'SLIDE'])
            self.next_step()
        else:
            self.finish()

    # ---------------------------------------------------------------- clues 3 and 4

    def next_step(self):
        """Start the next step of the clue 3 / 4 route (then land)."""
        if not self.route:
            self.finish()
            return
        state = self.route.popleft()
        self.hold_point = (self.x, self.y)
        if state in LEGS:
            prefix, heading = LEGS[state]
            gx, gy = self.points[prefix]
            speed = self.bottom_speed if state == 'BOTTOM' else self.leg_speed
            length = math.hypot(gx - self.x, gy - self.y)
            self.goal = (gx, gy, heading, speed, 8.0 + 2.0 * length / speed)
            self.goal_turned = False
        elif state == 'CLUE3':
            self.face_yaw = EAST
            self.seen_since = None
        elif state == 'SLIDE':
            self.slide_x = self.points['clue4_line'][0]     # the line itself (the plateau keeps x honest)
        self.set_state(state)

    def to_corner(self, dt, v_forward, v_left):
        """On up the west wall to corner_front from the north wall. True there."""
        if self.fresh(self.front_time):
            error = self.front - self.corner_front
            speed = max(-0.15, min(self.leg_speed, self.approach_gain * error))
        else:
            error, speed = None, 0.15
        self.follow_west_wall(self.west_wall_distance, speed, dt, v_forward, v_left)
        self.get_logger().info(f'TO_CORNER: front {self.front:.2f} m', throttle_duration_sec=1.0)
        return self.settled(error is not None and abs(error) < 0.08, 0.3)

    def fly_leg(self, dt):
        """Turn on the spot to the leg's heading, then fly to its end point.
        True once there."""
        gx, gy, heading, speed, _ = self.goal
        if not self.goal_turned:
            self.hold(self.hold_point, heading, dt)
            if self.settled(abs(wrap(heading - self.yaw)) < 0.08, 0.3):
                self.goal_turned = True
                self.settled_time = None
            return False
        ex, ey = gx - self.x, gy - self.y
        distance = math.hypot(ex, ey)
        if distance < 0.2:
            self.hold((gx, gy), heading, dt)
        else:
            # Velocity toward the end point, slowing down over the last metre
            v = min(speed, self.approach_gain * distance)
            ax = self.speed_kp * (v * ex / distance - self.vx)
            ay = self.speed_kp * (v * ey / distance - self.vy)
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            self.publish(c * ax + s * ay, -s * ax + c * ay, heading, dt)
        self.get_logger().info(f'{self.state}: {distance:.2f} m to go', throttle_duration_sec=1.0)
        return self.settled(distance < 0.08, 0.3)

    def slide(self, dt, elapsed):
        """Along the clue 4 line (x from the plateau), the right camera looking
        south-west. North until the board is in view, then servo on it: it
        moves from the right of the image to the left going north, so north
        while it's right of the centre, south while it's left. Hold once centred."""
        seen = self.servo_clue4(dt)
        if seen and self.settled(abs(self.right_board_bearing) < 0.06, 0.5):
            self.get_logger().info(f'Clue 4: in the middle of the right image ({self.right_board_area * 100:.1f}% '
                                   f'of it) at y {self.y:+.2f} - holding {self.clue4_dwell:.0f} s')
            self.set_state('CLUE4_HOLD')
        elif elapsed > 20.0 or (not seen and self.y > self.slide_max_y):
            if seen:
                self.set_state('CLUE4_HOLD')
            else:
                self.get_logger().warn(f'Clue 4: not seen by y {self.y:+.2f}')
                self.finish()

    def servo_clue4(self, dt):
        """Hold the clue 4 line, moving along it to centre the board in the
        right camera (north to search). True while the board is in view."""
        seen = self.board_in_view('right')
        if seen:
            vy = max(-self.slide_speed, min(self.slide_speed, self.slide_gain * self.right_board_bearing))
        else:
            vy = self.slide_speed if self.state == 'SLIDE' else 0.0
        ax = self.position_kp * (self.slide_x - self.x) - self.position_kd * self.vx
        ay = self.speed_kp * (vy - self.vy)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.publish(c * ax + s * ay, -s * ax + c * ay, self.clue4_heading, dt)
        return seen

    def finish(self):
        self.hold_point = (self.x, self.y)
        self.route.clear()
        self.set_state('DONE')

    def skip(self, state):
        # Watchdog gave up: a stuck takeoff rides anyway, a stuck ride or leg
        # goes on to the next step
        if state == 'TAKEOFF':
            self.start_ride()
        elif state == 'RIDE':
            self.after_clue2()
        elif state == 'SLIDE':
            self.set_state('CLUE4_HOLD') if self.board_in_view('right') else self.finish()
        elif state in ROUTE_STATES:
            self.next_step()
        else:
            self.finish()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = DroneWestRide()
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
