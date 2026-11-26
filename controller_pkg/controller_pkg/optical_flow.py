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
from collections import deque

import cv2
import numpy as np


class GroundVelocityEstimator:

    # Phase correlation response below this is treated as unreliable
    MIN_RESPONSE = 0.05
    # Gazebo sometimes re-publishes the previous render with a new timestamp.
    # A genuinely new frame differs by at least the camera noise (mean abs
    # difference ~2 grey levels), so anything below this is a repeat.
    MIN_FRAME_DIFFERENCE = 0.5

    def __init__(self, horizontal_fov_rad, image_width_px, smoothing=0.5,
                 crop_fraction=1.0, min_texture=0.0, downscale=0.5, baseline=0.0,
                 min_isotropy=0.0):
        """crop_fraction: use only this central fraction of the image (e.g. 0.5),
        so walls or cliffs near the edges of the wide view don't pull the
        estimate; the ground straight below is at the measured height.
        min_texture: frames whose (cropped) grey-level standard deviation is
        below this have nothing to track (e.g. open water) and are skipped.
        downscale: image scale used for the correlation (1.0 = full resolution).
        baseline: compare with the frame from about this many seconds ago
        instead of the previous one. High up, the ground moves well under a
        pixel between frames and the shift is lost in the noise; over a longer
        baseline it becomes several pixels (0 = previous frame).
        min_isotropy: when the texture runs in one direction only (a straight
        road edge on flat paint: smallest / largest eigenvalue of the gradient
        structure tensor below this), the shift along the lines can't be
        measured - phase correlation reports ~0 there. Then `constraint` is set
        to the (forward, left) unit vector across the lines, the only direction
        the estimate is good for (0 = never)."""
        self.tan_half_fov = math.tan(horizontal_fov_rad / 2.0)
        self.image_width_px = image_width_px
        self.smoothing = smoothing
        self.crop_fraction = crop_fraction
        self.min_texture = min_texture
        self.downscale = downscale
        self.baseline = baseline
        self.min_isotropy = min_isotropy
        self.constraint = None   # (forward, left) unit vector: the only measured direction
        self.history = deque(maxlen=20)   # (stamp, grey) for the long baseline
        self.prev = None
        self.prev_time = None
        self.window = None
        self.forward = 0.0   # m/s
        self.left = 0.0      # m/s
        self.valid = False

    def reset(self):
        self.history.clear()
        self.prev = None
        self.prev_time = None
        self.forward = self.left = 0.0
        self.valid = False

    def update(self, frame, stamp, height):
        """Update with a BGR frame, its time [s] and the camera height above
        the ground [m]. Returns (forward, left) velocity in m/s, or None if
        no reliable estimate is available yet."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.crop_fraction < 1.0:
            h, w = gray.shape
            ch, cw = int(h * self.crop_fraction), int(w * self.crop_fraction)
            gray = gray[(h - ch) // 2:(h + ch) // 2, (w - cw) // 2:(w + cw) // 2]
        if self.downscale != 1.0:
            gray = cv2.resize(gray, None, fx=self.downscale, fy=self.downscale,
                              interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.float32)
        if gray.std() < self.min_texture:
            # Nothing to track: drop the reference so the next textured frame starts fresh
            self.prev = None
            self.history.clear()
            self.valid = False
            return None
        self.constraint = self.measured_direction(gray) if self.min_isotropy > 0.0 else None
        if self.window is None or self.window.shape != gray.shape:
            self.window = cv2.createHanningWindow(gray.shape[::-1], cv2.CV_32F)

        prev, prev_time = self.prev, self.prev_time
        if prev is not None and float(np.abs(gray - prev).mean()) < self.MIN_FRAME_DIFFERENCE:
            # Repeated frame: keep the previous one and its time, report the last estimate
            return (self.forward, self.left) if self.valid else None
        self.prev, self.prev_time = gray, stamp
        if self.baseline > 0.0 and stamp is not None:
            # Long baseline: the stored frame closest to `baseline` s ago (at
            # least half that old)
            self.history.append((stamp, gray))
            old = [(t, g) for t, g in self.history if stamp - t >= 0.5 * self.baseline]
            if not old:
                self.valid = False
                return None
            prev_time, prev = min(old, key=lambda h: abs((stamp - h[0]) - self.baseline))
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
        scale = metres_per_px / self.downscale / dt
        forward = dy * scale
        left = dx * scale

        a = self.smoothing
        self.forward = a * self.forward + (1.0 - a) * forward
        self.left = a * self.left + (1.0 - a) * left
        self.valid = True
        return self.forward, self.left

    def measured_direction(self, gray):
        """None if the texture has gradients in every direction, else the
        (forward, left) unit vector across its lines."""
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        jxx, jxy, jyy = float(np.mean(gx * gx)), float(np.mean(gx * gy)), float(np.mean(gy * gy))
        values, vectors = np.linalg.eigh(np.array([[jxx, jxy], [jxy, jyy]]))
        if values[1] <= 0.0 or values[0] / values[1] >= self.min_isotropy:
            return None
        u, v = vectors[:, 1]      # strongest gradient (image x right, y down), across the lines
        # Image y is forward and image x is left (see update)
        return float(v), float(u)
