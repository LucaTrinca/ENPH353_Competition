"""Classical CV road detection + PID steering for the paved road.

The road is dark gray (HSV: S ~ 0, V ~ 67), the lines are white (V ~ 224)
and the grass is saturated green. The sky is also gray, so only the bottom
part of the image is used.
"""
import math

import cv2
import numpy as np


class PID:

    def __init__(self, kp, ki, kd, integral_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error, dt):
        if dt <= 0.0:
            dt = 1e-3
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class LaneFollower:

    # HSV thresholds for the gray road surface
    ROAD_LOWER = np.array([0, 0, 40])
    ROAD_UPPER = np.array([180, 40, 120])

    # Region of interest (fraction of image height) used to find the road
    ROI_TOP = 0.6
    ROI_BOTTOM = 1.0

    # Minimum fraction of ROI pixels that must be road to trust the estimate
    MIN_ROAD_FRACTION = 0.02

    def __init__(self):
        self.mask = None
        self.roi_top_px = 0
        self.centroid_x = None

    def compute_error(self, frame):
        """Return the normalized lateral error in [-1, 1] (positive = road is
        to the right of the image center), or None if no road is visible."""
        h, w = frame.shape[:2]
        self.roi_top_px = int(h * self.ROI_TOP)
        roi = frame[self.roi_top_px:int(h * self.ROI_BOTTOM), :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.ROAD_LOWER, self.ROAD_UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        self.mask = mask

        moments = cv2.moments(mask, binaryImage=True)
        if moments['m00'] < self.MIN_ROAD_FRACTION * mask.size:
            self.centroid_x = None
            return None

        self.centroid_x = moments['m10'] / moments['m00']
        return (self.centroid_x - w / 2.0) / (w / 2.0)

    def draw_debug(self, frame):
        """Overlay the road mask, ROI and centroid on the frame (in place)."""
        if self.mask is None:
            return
        h, w = frame.shape[:2]
        roi = frame[self.roi_top_px:, :]
        overlay = roi.copy()
        overlay[self.mask > 0] = (255, 0, 255)
        cv2.addWeighted(overlay, 0.35, roi, 0.65, 0, dst=roi)
        cv2.line(frame, (0, self.roi_top_px), (w, self.roi_top_px), (255, 255, 0), 1)
        cv2.line(frame, (w // 2, self.roi_top_px), (w // 2, h), (255, 255, 0), 1)
        if self.centroid_x is not None:
            cx = int(self.centroid_x)
            cv2.circle(frame, (cx, (self.roi_top_px + h) // 2), 8, (0, 0, 255), -1)


class DownwardRoadDetector:
    """Find the road under the drone in the downward-facing camera image.

    Image top = drone forward, image right = drone right, image centre =
    directly below the drone. Only the stretch of road under the drone is
    used, so other parts of the winding road in view don't pull the estimate:
      - lateral error: centre of the run of road pixels on the centre row
        that contains (or is closest to) the image centre
      - heading error: on a circle around the image centre, the road arcs
        connected to the road under the drone. The arc the drone came from
        (closest to the direction behind its motion) is dropped, and of the
        rest in the forward half (+/- FORWARD_HALF_ANGLE) the rightmost one
        is taken, so at junctions the drone always turns right.
    """

    # Tighter than the car's range: the arena walls' gray tree texture
    # (S ~ 50, V ~ 40) must not pass, the road is S ~ 0, V ~ 65-75
    ROAD_LOWER = np.array([0, 0, 50])
    ROAD_UPPER = np.array([180, 25, 90])

    # Look-ahead circle radius as a fraction of the image height
    LOOKAHEAD_FRACTION = 0.45
    FORWARD_HALF_ANGLE_DEG = 110
    SCAN_STEP_DEG = 2
    # An arc counts as "where the drone came from" if its centre is within
    # this angle of the backward direction
    CAME_FROM_TOLERANCE_DEG = 90
    # Arcs narrower than this many scan samples are ignored as noise
    MIN_ARC_SAMPLES = 3
    # Rows above/below the centre row used to find the road under the drone
    CENTER_BAND_HALF_HEIGHT = 4
    # A road run on the centre row only counts as "under the drone" if it is
    # within this fraction of the half-width of the image centre
    MAX_UNDER_DISTANCE_FRACTION = 0.3

    def __init__(self):
        self.mask = None
        self.under_run = None       # (x_start, x_end) of the road under the drone
        self.lookahead_point = None  # (x, y) on the circle the drone steers to
        self.lookahead_radius = 0
        self.lateral_error = None
        self.heading_error = None
        # Nearest road pixel to the image centre, (dx, dy) in px (dx right,
        # dy down), or None if no road is visible at all
        self.nearest_road = None

    @staticmethod
    def runs(values):
        """Return (start, end) index pairs of consecutive True values."""
        padded = np.concatenate(([False], values, [False])).astype(np.int8)
        edges = np.flatnonzero(np.diff(padded))
        return list(zip(edges[0::2], edges[1::2] - 1))

    @staticmethod
    def angle_diff(a, b):
        return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)

    def circular_arcs(self, on_road, angles):
        """Centre angles of the runs of True in a circular array."""
        if on_road.all():
            return [0.0]
        if not on_road.any():
            return []
        # Rotate so the array starts on a False sample (no run wraps around)
        k = int(np.flatnonzero(~on_road)[0])
        values = np.roll(on_road, -k)
        unwrapped = np.unwrap(np.roll(angles, -k))
        return [math.remainder((unwrapped[s] + unwrapped[e]) / 2.0, 2.0 * math.pi)
                for s, e in self.runs(values) if e - s + 1 >= self.MIN_ARC_SAMPLES]

    def detect(self, frame, back_angle=math.pi):
        """Return (lateral_error, heading_error) or (None, None) if no road.

        back_angle: direction [rad] the drone came from, in the same angle
                    convention as heading_error (pi = straight behind). Pass
                    the opposite of the direction of travel when it is known.
        lateral_error: road centre under the drone, normalized to [-1, 1],
                       positive = road is to the drone's right.
        heading_error: angle [rad] of the road ahead relative to the drone's
                       forward axis, positive = road turns right.
        """
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.ROAD_LOWER, self.ROAD_UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # Fill crosswalk stripes so they don't split the road
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        self.mask = mask

        points = cv2.findNonZero(mask)
        if points is None:
            self.nearest_road = None
        else:
            offsets = points.reshape(-1, 2) - np.array([cx, cy])
            self.nearest_road = tuple(offsets[np.argmin((offsets ** 2).sum(axis=1))])

        # Lateral: road run on the centre rows under (or nearest to) the drone
        b = self.CENTER_BAND_HALF_HEIGHT
        row = mask[int(cy) - b:int(cy) + b + 1, :].max(axis=0) > 0
        row_runs = self.runs(row)
        if not row_runs:
            self.under_run = self.lookahead_point = None
            self.lateral_error = self.heading_error = None
            return None, None
        def distance(r):
            return 0 if r[0] <= cx <= r[1] else min(abs(r[0] - cx), abs(r[1] - cx))
        self.under_run = min(row_runs, key=distance)
        if distance(self.under_run) > self.MAX_UNDER_DISTANCE_FRACTION * w / 2.0:
            self.under_run = self.lookahead_point = None
            self.lateral_error = self.heading_error = None
            return None, None
        under_x = (self.under_run[0] + self.under_run[1]) / 2.0
        self.lateral_error = (under_x - cx) / (w / 2.0)

        # Only road connected to the road under the drone within the look-ahead
        # circle counts, so other stretches of road in view are ignored
        radius = self.LOOKAHEAD_FRACTION * h
        self.lookahead_radius = int(radius)
        disk = np.zeros_like(mask)
        cv2.circle(disk, (int(cx), int(cy)), int(radius) + 2, 255, -1)
        _, labels = cv2.connectedComponents(cv2.bitwise_and(mask, disk))
        band_labels = labels[int(cy) - b:int(cy) + b + 1,
                             self.under_run[0]:self.under_run[1] + 1]
        band_labels = band_labels[band_labels > 0]
        own_label = np.bincount(band_labels).argmax() if band_labels.size else 0

        # Heading: road arcs on the whole look-ahead circle
        angles = np.radians(np.arange(-180, 180, self.SCAN_STEP_DEG))
        xs = np.round(cx + radius * np.sin(angles)).astype(int)
        ys = np.round(cy - radius * np.cos(angles)).astype(int)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        on_road = np.zeros(len(angles), dtype=bool)
        on_road[inside] = labels[ys[inside], xs[inside]] == own_label
        mids = self.circular_arcs(on_road, angles)

        # Drop the arc the drone came from, then take the rightmost forward arc
        if mids:
            came_from = min(mids, key=lambda m: self.angle_diff(m, back_angle))
            if self.angle_diff(came_from, back_angle) < math.radians(self.CAME_FROM_TOLERANCE_DEG):
                mids.remove(came_from)
        forward = [m for m in mids if abs(m) <= math.radians(self.FORWARD_HALF_ANGLE_DEG)]
        if not forward:
            # Road under the drone but nothing ahead (e.g. end of the road)
            self.lookahead_point = None
            self.heading_error = 0.0
            return self.lateral_error, self.heading_error
        self.heading_error = float(max(forward))
        self.lookahead_point = (int(cx + radius * np.sin(self.heading_error)),
                                int(cy - radius * np.cos(self.heading_error)))
        return self.lateral_error, self.heading_error

    def draw_debug(self, frame):
        """Overlay the road mask, the road under the drone and the look-ahead point."""
        if self.mask is None:
            return
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        overlay = frame.copy()
        overlay[self.mask > 0] = (255, 0, 255)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)
        cv2.circle(frame, center, self.lookahead_radius, (255, 255, 0), 1)
        cv2.drawMarker(frame, center, (255, 255, 0), cv2.MARKER_CROSS, 10, 1)
        if self.under_run is not None:
            cv2.line(frame, (int(self.under_run[0]), center[1]),
                     (int(self.under_run[1]), center[1]), (0, 255, 255), 2)
        if self.lookahead_point is not None:
            cv2.line(frame, center, self.lookahead_point, (0, 0, 255), 2)
            cv2.circle(frame, self.lookahead_point, 4, (0, 0, 255), -1)
