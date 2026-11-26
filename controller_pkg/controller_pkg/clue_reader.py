"""Read a clue banner: cut it into characters and classify each one.

Input: the perspective-corrected banner from sign_detector.flatten().

How the banner is made (competition clueboard_generator.py): the text is
drawn on clue_banner.png (600 x 400) in UbuntuMono-R at size 90, the clue
type at (250, 30) and the value at (30, 250). The font is monospaced (50.4 px
per character), so characters sit on a fixed grid - this reader uses that
instead of trusting contours alone:

  1. resize the banner to 600 x 400 and compute "blueness" = B - (G + R) / 2.
     The letters are the only blue: the background is grey at any brightness,
     the logo is red and black. This ignores the lighting completely.
  2. split it into the type line (top half) and the value line (bottom half)
  3. find the ink blobs of each line (connected components, like the team
     notebook's character cropping) and put each one in its grid slot, from
     its centre and the 50.4 px pitch. Blobs that span two slots (letters
     merged when the board is far away) are split; pieces in one slot (a
     letter broken up) are merged; empty slots are spaces.
  4. crop each character like the notebook (ink box + 4 px, 32 x 32) and
     classify it with the team notebook's CNN, trained on banners rendered
     exactly as the competition does and degraded to look like the drone
     camera (train_clue_reader.py), run from ONNX with OpenCV. The older
     HOG + SVM model still works (pass its .yml).

The clue type is always one of the competition's keywords, so it is snapped
to the closest one.
"""
import os

import cv2
import numpy as np

BANNER_W, BANNER_H = 600, 400
PITCH = 50.4                      # px per character (UbuntuMono at size 90)
TYPE_X, VALUE_X = 250, 30         # where the generator starts each line
TYPE_MAX, VALUE_MAX = 6, 12       # characters on the banner (a 12th is half cut off by its edge, e.g. FAME N GLORY)
CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'   # notebook label order (sorted)
KEYWORDS = ['SIZE', 'VICTIM', 'CRIME', 'TIME', 'PLACE', 'MOTIVE', 'WEAPON', 'BANDIT']
CROP_SIZE = 32
PAD = 4

_HOG = cv2.HOGDescriptor((CROP_SIZE, CROP_SIZE), (16, 16), (8, 8), (8, 8), 9)


def blueness(banner):
    """Banner (BGR, any size) -> 600 x 400 float image, 0 = background, ~1 = ink."""
    image = cv2.resize(banner, (BANNER_W, BANNER_H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    b, g, r = cv2.split(image)
    blue = np.clip(b - (g + r) / 2.0, 0, None)
    top = np.percentile(blue, 99.5)
    return blue / top if top > 10 else blue * 0


def segment_line(ink, y_range, x_start, max_chars, threshold=0.35, min_area=40):
    """Cut one text line into grid slots.

    Returns a list with one entry per slot from the first character: None for
    a space, else the (x0, y0, x1, y1) ink box of that character."""
    y0, y1 = y_range
    band = ink[y0:y1]
    mask = (band > threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    blobs = [(stats[i], centroids[i]) for i in range(1, count)
             if stats[i][cv2.CC_STAT_AREA] >= min_area
             and stats[i][cv2.CC_STAT_HEIGHT] >= 12]
    if not blobs:
        return []
    # The line's height: most characters span it, so take the median box
    tops = np.median([s[cv2.CC_STAT_TOP] for s, _ in blobs])
    heights = np.median([s[cv2.CC_STAT_HEIGHT] for s, _ in blobs])
    # Drop specks well outside the text band (logo bits, frame remnants), and
    # strips of the blue frame along the banner's edge: taller than the letters
    # or running into the top / bottom of the banner, which no letter reaches
    band_height = band.shape[0]
    blobs = [(s, c) for s, c in blobs
             if s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT] > tops + 0.3 * heights
             and s[cv2.CC_STAT_TOP] < tops + 0.7 * heights
             and s[cv2.CC_STAT_HEIGHT] < 1.3 * heights
             and s[cv2.CC_STAT_TOP] > 0
             and s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT] < band_height]
    if not blobs:
        return []

    # Grid: slot k covers [origin + k * PITCH, origin + (k + 1) * PITCH). The
    # generator's start is known, but the crop may be shifted a little, so the
    # origin is fitted to the blobs (each blob centre should be mid-slot)
    centres = np.array([c[0] for _, c in blobs])
    best_origin, best_cost = None, None
    for origin in np.arange(x_start - 30, x_start + 30, 1.0):
        phase = (centres - origin) / PITCH - 0.5
        cost = float(np.sum((phase - np.round(phase)) ** 2))
        if best_cost is None or cost < best_cost:
            best_origin, best_cost = origin, cost
    origin = best_origin

    slots = {}
    for stats, _ in blobs:
        x, y, w, h = (stats[cv2.CC_STAT_LEFT], stats[cv2.CC_STAT_TOP],
                      stats[cv2.CC_STAT_WIDTH], stats[cv2.CC_STAT_HEIGHT])
        if w <= 1.3 * PITCH:
            # One character (no character is much wider than the pitch): the
            # slot of its centre. Splitting at the boundaries cut a character in
            # two when the banner's flattening stretched one side a little
            first = last = int(np.floor((x + w / 2.0 - origin) / PITCH))
        else:
            # Touching characters: split at the slot boundaries
            first = int(np.floor((x + 0.25 * PITCH - origin) / PITCH))
            last = int(np.floor((x + w - 0.25 * PITCH - origin) / PITCH))
            last = max(first, last)
        for k in range(first, last + 1):
            # A blob over several slots is split at the slot boundaries
            sx0 = max(x, int(origin + k * PITCH)) if last > first else x
            sx1 = min(x + w, int(origin + (k + 1) * PITCH)) if last > first else x + w
            box = slots.get(k)
            new = (sx0, y, sx1, y + h)
            slots[k] = new if box is None else (min(box[0], new[0]), min(box[1], new[1]),
                                                max(box[2], new[2]), max(box[3], new[3]))
    if not slots:
        return []
    first, last = min(slots), max(slots)
    if first < 0:
        first = 0
    keys = list(range(first, min(last, first + max_chars - 1) + 1))
    # Clue text has single spaces only: ink after two empty slots in a row is
    # something else (an object in front of the board's edge, e.g. the truck)
    for i in range(1, len(keys) - 1):
        if keys[i] not in slots and keys[i + 1] not in slots:
            keys = keys[:i]
            break
    return [None if k not in slots else
            (slots[k][0], slots[k][1] + y0, slots[k][2], slots[k][3] + y0) for k in keys]


def crop_character(image, box):
    """Notebook-style crop: ink box + 4 px, resized to 32 x 32."""
    x0, y0, x1, y1 = box
    h, w = image.shape[:2]
    crop = image[max(0, y0 - PAD):min(h, y1 + PAD), max(0, x0 - PAD):min(w, x1 + PAD)]
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


def hog_features(crop):
    """crop: 32 x 32 blueness (0..1 float). Returns a 1-D float32 feature vector."""
    image = np.clip(crop * 255.0, 0, 255).astype(np.uint8)
    return _HOG.compute(image).reshape(-1)


def segment(banner):
    """Banner -> (ink image, [type slots], [value slots])."""
    ink = blueness(banner)
    type_slots = segment_line(ink, (0, BANNER_H // 2), TYPE_X, TYPE_MAX)
    value_slots = segment_line(ink, (BANNER_H // 2, BANNER_H), VALUE_X, VALUE_MAX)
    return ink, type_slots, value_slots


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def snap_keyword(text):
    """Closest competition keyword and its edit distance."""
    best = min(KEYWORDS, key=lambda k: edit_distance(text, k))
    return best, edit_distance(text, best)


class ClueReader:
    """Reads banners with the CNN (.onnx, default) or the older HOG + SVM (.yml)."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = default_model_path()
        self.model_path = model_path
        self.kind = 'onnx' if model_path.endswith('.onnx') else 'svm'
        if self.kind == 'onnx':
            self.net = cv2.dnn.readNetFromONNX(model_path)
        else:
            self.svm = cv2.ml.SVM_load(model_path)
            if self.svm.empty():
                raise RuntimeError(f'Could not load the character model {model_path}')

    def classify(self, ink, boxes):
        """Characters and confidences (0..1) for a list of boxes."""
        if not boxes:
            return [], []
        if self.kind == 'onnx':
            # CNN (team notebook architecture, train_clue_reader.py): 32 x 32
            # blueness crops (0..1), NCHW, softmax over CHARACTERS
            batch = np.stack([crop_character(ink, b) for b in boxes]).astype(np.float32)[:, None]
            self.net.setInput(batch)
            probs = self.net.forward()
            idx = probs.argmax(axis=1)
            return [CHARACTERS[i] for i in idx], [float(probs[k, i]) for k, i in enumerate(idx)]
        features = np.stack([hog_features(crop_character(ink, b)) for b in boxes]).astype(np.float32)
        _, labels = self.svm.predict(features)
        chars = [CHARACTERS[int(l)] for l in labels.ravel()]
        # The multi-class SVM gives no probabilities; reliability comes from
        # voting over frames (drone_wall_ride) and the keyword snap distance
        return chars, [1.0] * len(chars)

    def read_line(self, ink, slots):
        boxes = [s for s in slots if s is not None]
        chars, conf = self.classify(ink, boxes)
        text, it = [], iter(chars)
        for s in slots:
            text.append(' ' if s is None else next(it))
        return ''.join(text).strip(), (min(conf) if conf else 0.0)

    def read(self, banner):
        """Banner image -> dict(type, value, raw_type, type_distance, confidence)."""
        ink, type_slots, value_slots = segment(banner)
        raw_type, type_conf = self.read_line(ink, type_slots)
        value, value_conf = self.read_line(ink, value_slots)
        keyword, distance = snap_keyword(raw_type.replace(' ', ''))
        return {'type': keyword, 'value': value, 'raw_type': raw_type,
                'type_distance': distance, 'confidence': min(type_conf, value_conf),
                'location': KEYWORDS.index(keyword) + 1}


DEFAULT_MODEL = 'clue_chars_cnn.onnx'


def default_model_path(name=DEFAULT_MODEL):
    """The trained model installed with the package (or next to this file when run from source)."""
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', name)
    if os.path.exists(here):
        return here
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('controller_pkg'), 'models', name)
    except Exception:  # noqa: BLE001
        return here
