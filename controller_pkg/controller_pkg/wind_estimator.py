"""Virtual wind sensor: wind estimated from the IMU and the drone's own commands.

Gazebo has no anemometer sensor and the wind plugin doesn't publish the gust
it applies, so the wind is measured from its effect on the drone. The IMU
accelerometer measures the specific force, i.e. every non-gravity force on
the drone divided by its mass. The drone knows the force it commanded (plus
the plugin's gravity compensation), so whatever is left over is the air force:

    F_air = m * a_imu - F_cmd - (0, 0, m * g)

Gazebo's wind model applies F_air = k * (wind - v_drone) with
k = force_approximation_scaling_factor (1 in fast_empty.sdf). F_air therefore
also contains the drone's own air drag (-k * v_drone). Adding back the
measured ground velocity separates out the wind itself:

    wind = F_air / k + v_drone

Everything is in the drone's body frame (x forward, y left); the plugin keeps
the drone level, so the IMU axes are the body axes. Only the horizontal
components are estimated, the competition wind is horizontal.
"""
import math

GRAVITY = 9.8


class WindEstimator:

    def __init__(self, mass, drag_coeff=1.0, filter_tau=0.3):
        self.mass = mass
        self.drag_coeff = drag_coeff
        self.filter_tau = filter_tau
        self.air_force = None   # (forward, left) in N, low-pass filtered
        self.last_time = None

    def reset(self):
        self.air_force = None
        self.last_time = None

    def update(self, accel, applied_force, stamp):
        """accel: IMU linear acceleration (x, y, z) in m/s^2.
        applied_force: force command (forward, left, up) in N sent to the drone.
        stamp: IMU message time in s."""
        force = (self.mass * accel[0] - applied_force[0],
                 self.mass * accel[1] - applied_force[1])
        if self.air_force is None or self.last_time is None:
            self.air_force = force
        else:
            dt = max(0.0, stamp - self.last_time)
            alpha = dt / (self.filter_tau + dt) if self.filter_tau > 0.0 else 1.0
            self.air_force = tuple(a + alpha * (f - a) for a, f in zip(self.air_force, force))
        self.last_time = stamp

    def wind_velocity(self, ground_velocity):
        """Wind velocity (forward, left) in m/s, given the drone's ground
        velocity (forward, left) in m/s, or None before the first update."""
        if self.air_force is None:
            return None
        return (self.air_force[0] / self.drag_coeff + ground_velocity[0],
                self.air_force[1] / self.drag_coeff + ground_velocity[1])

    def feedforward(self, ground_velocity):
        """Force (forward, left) in N that cancels the wind's push but keeps
        the drone's own air drag, or (0, 0) before the first update."""
        wind = self.wind_velocity(ground_velocity)
        if wind is None:
            return 0.0, 0.0
        return -self.drag_coeff * wind[0], -self.drag_coeff * wind[1]

    @staticmethod
    def speed(wind):
        return math.hypot(*wind) if wind is not None else 0.0
