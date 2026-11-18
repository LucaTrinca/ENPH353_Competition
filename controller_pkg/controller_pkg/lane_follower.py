"""Classical CV road detection + PID steering for the paved road.

The road is dark gray (HSV: S ~ 0, V ~ 67), the lines are white (V ~ 224)
and the grass is saturated green. The sky is also gray, so only the bottom
part of the image is used.
"""
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
