"""Ground velocity from the downward-facing camera (optical flow).

Phase correlation between consecutive frames gives the image shift of the
ground in pixels. With the camera pointing straight down at flat ground the
image is a uniformly scaled copy of the ground, so the shift converts to
metres using the height above the ground and the camera field of view.

Image top = drone forward, image right = drone right. When the drone moves
forward the ground moves down the image (+y); when it moves left the ground
moves right in the image (+x).
"""
import math

import cv2
import numpy as np


class GroundVelocityEstimator:

    # Phase correlation response below this is treated as unreliable
    MIN_RESPONSE = 0.05
    DOWNSCALE = 0.5
    # Gazebo sometimes re-publishes the previous render with a new timestamp.
    # A genuinely new frame differs by at least the camera noise (mean abs
    # difference ~2 grey levels), so anything below this is a repeat.
    MIN_FRAME_DIFFERENCE = 0.5

    def __init__(self, horizontal_fov_rad, image_width_px, smoothing=0.5):
        self.tan_half_fov = math.tan(horizontal_fov_rad / 2.0)
        self.image_width_px = image_width_px
        self.smoothing = smoothing
        self.prev = None
        self.prev_time = None
        self.window = None
        self.forward = 0.0   # m/s
        self.left = 0.0      # m/s
        self.valid = False

    def reset(self):
        self.prev = None
        self.prev_time = None
        self.forward = self.left = 0.0
        self.valid = False

    def update(self, frame, stamp, height):
        """Update with a BGR frame, its time [s] and the camera height above
        the ground [m]. Returns (forward, left) velocity in m/s, or None if
        no reliable estimate is available yet."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=self.DOWNSCALE, fy=self.DOWNSCALE,
                          interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.window is None or self.window.shape != gray.shape:
            self.window = cv2.createHanningWindow(gray.shape[::-1], cv2.CV_32F)

        prev, prev_time = self.prev, self.prev_time
        if prev is not None and float(np.abs(gray - prev).mean()) < self.MIN_FRAME_DIFFERENCE:
            # Repeated frame: keep the previous one and its time, report the last estimate
            return (self.forward, self.left) if self.valid else None
        self.prev, self.prev_time = gray, stamp
        if prev is None or stamp is None or prev_time is None or height is None:
            self.valid = False
            return None
        dt = stamp - prev_time
        if dt <= 0.0 or height <= 0.0:
            self.valid = False
            return None

        (dx, dy), response = cv2.phaseCorrelate(prev, gray, self.window)
        if response < self.MIN_RESPONSE:
            self.valid = False
            return None

        # Metres per (full-resolution) pixel on the ground
        metres_per_px = 2.0 * height * self.tan_half_fov / self.image_width_px
        scale = metres_per_px / self.DOWNSCALE / dt
        forward = dy * scale
        left = dx * scale

        a = self.smoothing
        self.forward = a * self.forward + (1.0 - a) * forward
        self.left = a * self.left + (1.0 - a) * left
        self.valid = True
        return self.forward, self.left
