"""IMU helpers."""
import math


def level_acceleration(msg):
    """Horizontal acceleration [m/s^2] in the drone's yaw-aligned (level) frame.

    The accelerometer measures specific force in the body frame. When the body
    is tilted even slightly, part of gravity shows up on the body x/y axes
    (0.5 deg of tilt reads as ~0.09 m/s^2, i.e. ~0.4 m of fake motion in 3 s).
    Rotating by the IMU's roll and pitch (not its yaw) removes that, leaving
    the real horizontal acceleration along the drone's forward / left axes.
    Returns (forward, left)."""
    q = msg.orientation
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    a = msg.linear_acceleration
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # f_level = Ry(pitch) Rx(roll) f_body
    y1 = cr * a.y - sr * a.z
    z1 = sr * a.y + cr * a.z
    forward = cp * a.x + sp * z1
    return forward, y1
