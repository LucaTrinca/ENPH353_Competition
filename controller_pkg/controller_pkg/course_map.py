"""Clue board locations and the hardcoded sign tour.

Sign spawn rules copied from the competition launch file
(2026_competition/src/my_robot_bringup/launch/my_robot.launch.py). World
frame: x along the long side of the arena (-6..6 m), y across it
(-2.75..2.75 m), z up. Each board also gets a random yaw of +/-0.15 rad.

A board is 0.2 x 0.16 m; its clue banner (0.15 x 0.1 m, centred 0.06 m above
the board's base) is on the board's local +y face, so a board with yaw
`yaw` faces the world direction (-sin(yaw), cos(yaw)).

Using this map is planning, not sensing: the drone only knows where it is
by dead reckoning from its known start pose.
"""
import math

# Drone spawn pose (competition launch file): x, y, yaw
START = (-5.5, -2.5, 1.57)

# (clue number, x or (min, max), y or (min, max), base z, yaw before the random +/-0.15)
SIGNS = [
    (1, -5.81, -1.64, 0.04, 3.14),
    (2, -5.16, (1.11, 1.6), 0.04, 3.14),
    (3, -4.0, 1.67, 0.04, 1.57),
    (4, -0.83, (0.0, 0.63), 0.04, 0.0),
    (5, -0.83, (-1.84, -1.44), 0.04, 3.14),
    (6, (3.0, 3.41), -1.71, 0.04, 1.57),
    (7, 3.8, 2.01, 0.04, -1.57),
    (8, 0.9, 1.2, 1.86, -1.57),  # on top of the mountain
]

# Height of the banner centre above the board's base [m]
BANNER_HEIGHT = 0.06


def middle(value):
    return (value[0] + value[1]) / 2.0 if isinstance(value, tuple) else value


def sign_position(sign):
    """Expected (x, y, z, yaw) of a sign; random coordinates use the middle of their range."""
    _, x, y, z, yaw = sign
    return middle(x), middle(y), z, yaw


def view_point(sign, distance):
    """(x, y, look_yaw): a point `distance` m in front of the banner, and the
    yaw that looks straight at it."""
    x, y, _, yaw = sign_position(sign)
    fx, fy = -math.sin(yaw), math.cos(yaw)
    return x + distance * fx, y + distance * fy, math.atan2(-fy, -fx)


# Route step: (label, x, y, altitude, altitude mode, yaw or None, dwell [s], photo)
#   altitude mode 'agl' = height above the ground (downward range sensor),
#                 'abs' = height above the start point (altimeter)
#   The drone flies straight to (x, y) at cruise speed, reaching the altitude
#   and yaw on the way, then holds there for `dwell` seconds.
# Legs that need a different transit altitude: the lake crossing to clue 6 is
# flown higher so the (cropped) down camera keeps the shore in view
TRANSIT_ALTITUDE_OVERRIDES = {6: 0.6}

# Extra waypoints flown before a clue's approach, (x, y): the straight line from
# clue 6 to clue 7 runs along the mountain's 1.8 m east cliff, which ruins the
# optical flow, so go around it at least ~1 m away
DETOURS = {7: [(3.9, -1.3)]}


def sign_tour(view_distance=0.35, transit_altitude=0.4, view_altitude=0.4,
              mountain_altitude=2.3, dwell=2.0):
    steps = [('takeoff', START[0], START[1], transit_altitude, 'agl', START[2], 1.0, False)]
    for sign in SIGNS:
        clue = sign[0]
        x, y, look = view_point(sign, view_distance)
        if clue == 8:
            # The mountain's sides are cliffs, so climb on the spot first, cross
            # high, and only follow the ground again once over the plateau
            px, py = steps[-1][1], steps[-1][2]
            steps.append(('climb over mountain', px, py, mountain_altitude, 'abs', None, 0.5, False))
            steps.append((f'approach clue {clue}', x, y, mountain_altitude, 'abs', look, 1.0, False))
        else:
            leg_altitude = TRANSIT_ALTITUDE_OVERRIDES.get(clue, transit_altitude)
            for i, (dx, dy) in enumerate(DETOURS.get(clue, [])):
                steps.append((f'detour {i + 1} to clue {clue}', dx, dy, leg_altitude, 'agl',
                              None, 0.0, False))
            steps.append((f'approach clue {clue}', x, y, leg_altitude, 'agl', look, 1.0, False))
        steps.append((f'view clue {clue}', x, y, view_altitude, 'agl', look, dwell, True))
        if clue != 8:
            steps.append((f'leave clue {clue}', x, y, transit_altitude, 'agl', look, 0.0, False))
    return steps


if __name__ == '__main__':
    print('clue  board x, y, z (random ranges)            faces     view point     look yaw')
    for sign in SIGNS:
        clue, x, y, z, yaw = sign
        fx, fy = -math.sin(yaw), math.cos(yaw)
        vx, vy, look = view_point(sign, 0.35)
        print(f'{clue:4d}  x {str(x):14s} y {str(y):14s} z {z:.2f}  ({fx:+.0f},{fy:+.0f})  '
              f'({vx:+.2f},{vy:+.2f})  {math.degrees(look):+5.0f} deg')
