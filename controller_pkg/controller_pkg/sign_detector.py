"""Clue board finder: detect a complete blue clue board in a camera image.

Based on extract_and_flatten_sign() from the team's character-recognition
notebook (blue HSV mask -> largest contour -> 4 corners -> perspective warp to
800 x 400 -> crop the border off), with these changes so it only reports a
board when the WHOLE board is in view and readable:

  - the notebook's 5 x 5 mask cleanup is tried first, then a finer 3 x 3 one
    that keeps the 1 - 3 px frame of a board ~2 m away (seen from the walls)
  - every blue contour is considered, largest first, not just the largest one
    (the lake or a distant blue object no longer hides a real board)
  - the board must not touch the image border - a board cut off by the edge
    of the image is rejected instead of being warped from a wrong box
  - it must be a clean convex quadrilateral (no minAreaRect fallback, which
    turns any blob into a box), that fills its outline, with a believable
    side-length ratio, and big enough to read. If a thin bit of blue-ish
    background touches the board and spoils the outline, the board is
    retried after an opening sized to the board
  - the inside of the quad must be mostly NOT blue (the white banner covers
    ~47% of the board face), which rejects the plain blue back of a board
  - corners are ordered by angle around the centre, which also works when
    the board is strongly rotated in the image

detect() returns a SignDetection (corners, area, a score) or None, and
flatten() warps the board like the notebook, then crops to the banner found
in the warped board (not a fixed border) at the notebook's crop size
(608 x 280), so no strips of the blue frame are left for the character mask.
"""
from dataclasses import dataclass

import cv2
import numpy as np

# Notebook values
LOWER_BLUE = np.array([90, 40, 50])
UPPER_BLUE = np.array([150, 255, 255])
FLAT_WIDTH, FLAT_HEIGHT = 800, 400
CROP_X, CROP_Y = 0.12, 0.15
BANNER_INSET = 2                  # px trimmed inside the frame (of the 800 x 400 warp)
# Output size of flatten() - the notebook's crop of its 800 x 400 warp
BANNER_SIZE = (FLAT_WIDTH - 2 * int(FLAT_WIDTH * CROP_X), FLAT_HEIGHT - 2 * int(FLAT_HEIGHT * CROP_Y))


@dataclass
class SignDetection:
    corners: np.ndarray      # 4 x 2 float32, top-left, top-right, bottom-right, bottom-left
    area: float              # px^2, quad area
    area_fraction: float     # of the whole image
    banner_fraction: float   # non-blue fraction inside the quad
    centre: tuple            # (x, y) px

    @property
    def score(self):
        """Bigger and more front-on boards score higher (for keeping the best view)."""
        return self.area * min(1.0, self.banner_fraction / 0.4)


def order_corners(pts):
    """Top-left, top-right, bottom-right, bottom-left, by angle around the centre."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(angles)]          # clockwise in image coordinates (y down)
    start = int(np.argmin(pts.sum(axis=1)))  # top-left = smallest x + y
    return np.roll(pts, -start, axis=0)


def quad_from_contour(contour):
    """The contour's 4 ordered corners if it is a convex quadrilateral, else None."""
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None
    return order_corners(approx)


def blue_mask(image, fine=False):
    """Blue pixels, cleaned up.

    Default: the notebook's 5 x 5 opening then closing - best for boards
    seen up close. fine=True: only a 3 x 3 closing, for boards ~2 m away whose
    frame bars are just 1 - 3 px thick (any opening erases them, and gaps in
    them break the board's outline)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    if fine:
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def detect(image, min_area_fraction=0.002, border_margin=4, min_banner_fraction=0.25,
           max_banner_fraction=0.85, max_side_ratio=3.5, min_fill=0.85, min_frame_blue=0.6):
    """Find the best complete clue board in a BGR image, or None.

    Tries the notebook's mask cleanup first, then a finer one that keeps the
    thin frame of a distant board."""
    for fine in (False, True):
        best = find_board(image, blue_mask(image, fine), min_area_fraction, border_margin,
                          min_banner_fraction, max_banner_fraction, max_side_ratio, min_fill,
                          min_frame_blue)
        if best is not None:
            return best
    return None


def find_board(image, mask, min_area_fraction, border_margin, min_banner_fraction,
               max_banner_fraction, max_side_ratio, min_fill, min_frame_blue):
    """The best complete board among the blue regions of `mask`, or None."""
    height, width = image.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_fraction * width * height

    best = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            break  # sorted, the rest are smaller

        # Whole board in view: nothing within the margin of the image border
        x, y, w, h = cv2.boundingRect(contour)
        if (x <= border_margin or y <= border_margin
                or x + w >= width - border_margin or y + h >= height - border_margin):
            continue

        detection = check_board(image, contour, min_banner_fraction, max_banner_fraction,
                                max_side_ratio, min_fill, min_frame_blue)
        if detection is None:
            # Something thin touching the board (a blue-ish bit of background,
            # a neighbouring object) can spoil the outline - open the mask with
            # a kernel the board's thick frame survives but slivers don't, and
            # try the largest remaining piece again
            # (smallest kernel first - a big one rounds the frame's corners)
            for scale in (0.03, 0.05, 0.08):
                k = max(5, int(scale * min(w, h))) | 1
                roi = cv2.morphologyEx(mask[y:y + h, x:x + w], cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
                pieces, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not pieces:
                    break
                piece = max(pieces, key=cv2.contourArea) + np.array([x, y], np.int32)
                if cv2.contourArea(piece) < min_area:
                    break
                detection = check_board(image, piece, min_banner_fraction, max_banner_fraction,
                                        max_side_ratio, min_fill, min_frame_blue)
                if detection is not None:
                    break
        if detection is not None and (best is None or detection.score > best.score):
            best = detection
    return best


def check_board(image, contour, min_banner_fraction, max_banner_fraction, max_side_ratio,
                min_fill, min_frame_blue):
    """SignDetection if the contour is a complete, clean clue board, else None."""
    height, width = image.shape[:2]
    corners = quad_from_contour(contour)
    if corners is None:
        return None
    quad_area = cv2.contourArea(corners)
    if quad_area <= 0 or cv2.contourArea(contour) / quad_area < min_fill:
        return None
    sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    if sides.min() <= 0 or sides.max() / sides.min() > max_side_ratio:
        return None

    # The banner: the inside of the board should be largely not blue
    inside = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(inside, corners.astype(np.int32), 255)
    inside = cv2.erode(inside, np.ones((3, 3), np.uint8))
    n_inside = cv2.countNonZero(inside)
    if n_inside == 0:
        return None
    banner = 1.0 - cv2.countNonZero(cv2.bitwise_and(mask_raw(image), inside)) / n_inside
    if not (min_banner_fraction <= banner <= max_banner_fraction):
        return None

    # A real board has a solid blue frame on all four sides. A quad that
    # swallowed a neighbouring blue object (or cut a corner) has a side whose
    # border strip is largely background.
    if min_frame_side(image, corners) < min_frame_blue:
        return None
    return SignDetection(corners, float(quad_area), float(quad_area / (width * height)),
                         float(banner), tuple(corners.mean(axis=0)))


def min_frame_side(image, corners, size=(200, 160), strip=0.05):
    """Blue fraction of the least-blue of the board's 4 border strips (0..1).

    The frame is 12.5% of the board's width at the sides and ~19% of its
    height at the top and bottom, so 5% strips lie well inside it."""
    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    mask = mask_raw(cv2.warpPerspective(image, matrix, (w, h))) > 0
    sx, sy = max(1, int(strip * w)), max(1, int(strip * h))
    return float(min(mask[:sy].mean(), mask[-sy:].mean(), mask[:, :sx].mean(), mask[:, -sx:].mean()))


def mask_raw(image):
    """Blue mask without the closing, so the banner's white area stays open."""
    return cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), LOWER_BLUE, UPPER_BLUE)


def flatten(image, detection, fit_banner=True):
    """The clue banner, perspective-corrected, at BANNER_SIZE (the notebook's crop size).

    Warps the board to 800 x 400 as the notebook does. The notebook then crops
    a fixed 12% / 15% border, which leaves strips of the blue frame when the
    board is seen at an angle, and the blue character mask picks those up as
    extra "characters". With fit_banner the crop is found from the warped
    board instead: the frame is where rows / columns are mostly blue."""
    dst = np.array([[0, 0], [FLAT_WIDTH - 1, 0], [FLAT_WIDTH - 1, FLAT_HEIGHT - 1],
                    [0, FLAT_HEIGHT - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(detection.corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, matrix, (FLAT_WIDTH, FLAT_HEIGHT))
    box = banner_box(warped) if fit_banner else None
    if box is None:
        cx, cy = int(FLAT_WIDTH * CROP_X), int(FLAT_HEIGHT * CROP_Y)
        box = (cx, cy, FLAT_WIDTH - 2 * cx, FLAT_HEIGHT - 2 * cy)
    x, y, w, h = box
    return cv2.resize(warped[y:y + h, x:x + w], BANNER_SIZE, interpolation=cv2.INTER_AREA)


def banner_box(warped, frame_fraction=0.8, inset=None):
    """(x, y, w, h) of the banner inside a warped board, or None.

    A row (column) belongs to the frame when more than frame_fraction of it is
    blue, measured over the middle 60% of the board, where the frame band is
    straight even if the warp is slightly off. The banner is the longest run
    of non-frame rows (columns). The frame is solid blue, while even a row
    through a long word (tops of SABOTAGE) has gaps between the letters."""
    if inset is None:
        inset = BANNER_INSET
    mask = mask_raw(warped) > 0
    height, width = mask.shape
    rows = mask[:, int(0.2 * width):int(0.8 * width)].mean(axis=1) < frame_fraction
    cols = mask[int(0.2 * height):int(0.8 * height), :].mean(axis=0) < frame_fraction

    def longest_run(flags):
        best, start = (0, 0), None
        for i, flag in enumerate(np.append(flags, False)):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start > best[1] - best[0]:
                    best = (start, i)
                start = None
        return best

    y0, y1 = longest_run(rows)
    x0, x1 = longest_run(cols)
    if (y1 - y0) < 0.4 * height or (x1 - x0) < 0.5 * width:
        return None
    return x0 + inset, y0 + inset, (x1 - x0) - 2 * inset, (y1 - y0) - 2 * inset


def draw(image, detection, label=None):
    """Draw a detection on an image (in place)."""
    cv2.polylines(image, [detection.corners.astype(np.int32)], True, (0, 255, 255), 2)
    if label:
        x, y = detection.corners[0]
        cv2.putText(image, label, (int(x), max(12, int(y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
