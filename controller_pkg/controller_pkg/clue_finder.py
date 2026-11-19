#!/usr/bin/env python3
"""Clue finder: find, read and submit clue boards seen by the drone's cameras.

Runs as its own process next to a flight controller (drone_wall_ride), so the
camera work can never slow the flight control down, and its window runs on
the main thread (OpenCV's Qt windows hang when used from any other thread).

  - looks for complete clue boards in the left / front / right cameras with
    sign_detector (the team notebook's clue finder, improved), on a half-size
    copy of the 1280 x 960 side images
  - while a board is in view, keeps its 5 biggest views; once it has been out
    of view for sign_lost_time, reads them with clue_reader and votes (bigger
    views count more; only readings whose clue type is within one letter of a
    keyword count). Views whose letters are under min_letter_px (a board far
    away) aren't read at all - they'd only add a wrong guess - so a board seen
    only from far away is logged but not submitted
  - logs the reading, saves the best frame and the flattened banner to
    <photo_dir>/clues_<time>/ with a clues_read.csv table, and (submit_clues)
    sends it to /score_tracker - location = the clue type's number (SIZE 1 ...
    BANDIT 8) - again only if a later view changes the answer
  - a board seen again (within 1 m of an earlier capture, or with the same
    clue type) counts as the same board

The flight controller's state (/drone/state) says when to look: only while it
is flying its route (active_states), or always if no state is ever received (e.g.
with the keyboard teleop). When the state becomes DONE, boards still in view
are read and submitted, then /drone/clues_flushed is published so the
controller can stop the timer after the last clue.

Every complete board it sees is also published on /drone/board_bearing (its
direction from the camera), which the flight controller uses to correct its
position on boards at known places.

Topics: /left_camera/image_raw, /camera/image_raw, /right_camera/image_raw,
        /drone/state, /drone/status, /drone/pose_estimate (sub),
        /score_tracker, /drone/clues_flushed, /drone/board_bearing,
        /drone/board_identified (pub).
"""
import math
import os
import signal
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String

from controller_pkg import clue_reader, sign_detector

WINDOW = 'Drone Cameras (left | front | right)'
CAMERAS = [('left', '/left_camera/image_raw'), ('front', '/camera/image_raw'),
           ('right', '/right_camera/image_raw')]
# Horizontal field of view of each camera [rad] (drone_gazebo.xacro)
CAMERA_FOV = {'left': math.radians(70.0), 'front': math.radians(60.0), 'right': math.radians(70.0)}


def detect_scaled(image, width):
    """sign_detector.detect on a downscaled copy, corners mapped back to full size."""
    scale = image.shape[1] / float(width)
    if scale <= 1.05:
        return sign_detector.detect(image)
    small = cv2.resize(image, (width, int(round(image.shape[0] / scale))), interpolation=cv2.INTER_AREA)
    detection = sign_detector.detect(small)
    if detection is None:
        return None
    corners = (detection.corners * scale).astype(np.float32)
    return sign_detector.SignDetection(corners, detection.area * scale * scale, detection.area_fraction,
                                       detection.banner_fraction, tuple(corners.mean(axis=0)))


def letter_height(detection):
    """Approximate height of the banner letters in the image [px]: they are
    about a tenth of the board's height (16 mm on a 0.16 m board)."""
    c = detection.corners
    board = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2.0
    return 0.1 * board


class ClueFinder(Node):

    def __init__(self):
        super().__init__('clue_finder')
        self.declare_parameter('team_id', 'TeamFizz')
        self.declare_parameter('team_password', 'pass1234')
        self.declare_parameter('show_cameras', True)        # left | front | right window
        self.declare_parameter('rate', 5.0)                 # Hz, camera processing
        self.declare_parameter('detect_width', 720)         # px, images are shrunk to this to find boards
        self.declare_parameter('sign_lost_time', 1.0)       # s out of view before a board is read
        self.declare_parameter('views_to_read', 5)          # biggest views of each board that are read
        self.declare_parameter('min_letter_px', 6.0)        # views with smaller letters are not read (garbage)
        self.declare_parameter('read_clues', True)
        self.declare_parameter('clue_model', '')            # '' = trained SVM, or the team CNN as .onnx
        self.declare_parameter('submit_clues', True)        # send readings to /score_tracker
        # Controller states to look for boards in (not on the ground, not once done)
        # (TAKEOFF too: clue 1 faces the spawn, the front camera sees it while climbing)
        self.declare_parameter('active_states', ['TAKEOFF', 'RIDE', 'TURN', 'BACK_OFF', 'CLIMB',
                                                 'CROSS', 'CLUE8', 'CLUE8_HOLD'])
        self.declare_parameter('record_interval', 0.0)      # s between saved raw frames, 0 = off
        self.declare_parameter('photo_dir', '~/drone_images')

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.team_id = p('team_id')
        self.team_password = p('team_password')
        self.show_cameras = bool(p('show_cameras'))
        self.period = 1.0 / float(p('rate'))
        self.detect_width = int(p('detect_width'))
        self.sign_lost_time = float(p('sign_lost_time'))
        self.views_to_read = int(p('views_to_read'))
        self.min_letter_px = float(p('min_letter_px'))
        self.submit_clues = bool(p('submit_clues'))
        self.active_states = list(p('active_states'))
        self.record_interval = float(p('record_interval'))
        self.folder = os.path.join(os.path.expanduser(p('photo_dir')),
                                   datetime.now().strftime('clues_%Y%m%d_%H%M%S'))
        self.reader = None
        if bool(p('read_clues')):
            try:
                self.reader = clue_reader.ClueReader(p('clue_model') or None)
                self.get_logger().info(f'Clue reader: {self.reader.kind} model {self.reader.model_path}')
            except Exception as error:  # noqa: BLE001
                self.get_logger().error(f'Clue reader disabled - {error}')

        self.bridge = CvBridge()
        self.score_pub = self.create_publisher(String, '/score_tracker', 10)
        self.flushed_pub = self.create_publisher(Empty, '/drone/clues_flushed', 1)
        # Direction of every complete board seen (the flight controller uses
        # boards at known places to correct its position)
        self.bearing_pub = self.create_publisher(PointStamped, '/drone/board_bearing', 10)
        # Each board read: "camera,location,TYPE,VALUE" (the flight controller
        # uses known boards as landmarks)
        self.identified_pub = self.create_publisher(String, '/drone/board_identified', 10)
        self.frames = {}
        for name, topic in CAMERAS:
            self.create_subscription(Image, topic, self.make_camera_callback(name), 1)
        self.create_subscription(String, '/drone/state', self.state_callback, 10)
        self.create_subscription(String, '/drone/status', self.status_callback, 1)
        self.create_subscription(PoseStamped, '/drone/pose_estimate', self.pose_callback, 1)

        self.state = None           # controller state, None = never heard (look all the time)
        self.status = ''
        self.pose = (0.0, 0.0)
        self.flushed = False
        self.events = {}            # camera -> board in view: views, best view, pose, last seen
        self.boards = []            # saved boards
        self.submitted = {}         # location -> value sent
        self.latest_detections = {}
        self.last_record = None
        self.get_logger().info(f'Clue finder started, saving to {self.folder}')

    # ---------------------------------------------------------------- inputs

    def make_camera_callback(self, name):
        def callback(msg):
            self.frames[name] = msg
        return callback

    def state_callback(self, msg):
        self.state = msg.data

    def status_callback(self, msg):
        self.status = msg.data

    def pose_callback(self, msg):
        self.pose = (msg.pose.position.x, msg.pose.position.y)

    def looking(self):
        return self.state is None or self.state in self.active_states

    # ---------------------------------------------------------------- work

    def step(self):
        """Process the latest frame of each camera. Returns the window image (or None)."""
        if not self.frames:
            return None
        stamps = [m.header.stamp.sec + m.header.stamp.nanosec / 1e9 for m in self.frames.values()]
        now = max(stamps)
        images = {name: self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                  for name, msg in list(self.frames.items())}
        if self.looking():
            self.find_boards(images, now)
        if self.state == 'DONE' and not self.flushed:
            # The flight is over: read and submit boards still in view, then
            # tell the controller it can stop the timer
            self.flush()
        if (self.record_interval > 0.0
                and (self.last_record is None or now - self.last_record >= self.record_interval)):
            self.last_record = now
            os.makedirs(os.path.join(self.folder, 'frames'), exist_ok=True)
            for name, image in images.items():
                cv2.imwrite(os.path.join(self.folder, 'frames', f'{now:08.1f}_{name}.png'), image)
        return self.build_window(images) if self.show_cameras else None

    def flush(self):
        for camera in list(self.events):
            self.save_board(camera, self.events.pop(camera))
        self.flushed = True
        self.get_logger().info(f'Clue boards captured: {len(self.boards)}')
        self.flushed_pub.publish(Empty())

    def publish_bearing(self, name, image, detection):
        """Board direction in the camera: x = bearing [rad] (positive = toward the
        image's right), y = fraction of the image the board covers. Stamped with
        the image time, frame_id = camera name."""
        width = image.shape[1]
        focal = (width / 2.0) / math.tan(CAMERA_FOV[name] / 2.0)
        msg = PointStamped()
        msg.header.stamp = self.frames[name].header.stamp
        msg.header.frame_id = name
        msg.point.x = math.atan((detection.centre[0] - width / 2.0) / focal)
        msg.point.y = float(detection.area_fraction)
        self.bearing_pub.publish(msg)

    def find_boards(self, images, now):
        for name, image in images.items():
            detection = detect_scaled(image, self.detect_width)
            self.latest_detections[name] = detection
            event = self.events.get(name)
            if detection is not None:
                self.publish_bearing(name, image, detection)
                if event is None:
                    event = self.events[name] = {'score': -1.0, 'views': []}
                    self.get_logger().info(f'Clue board in view of the {name} camera '
                                           f'(estimate x {self.pose[0]:+.2f} y {self.pose[1]:+.2f})')
                event['last_seen'] = now
                views = event['views']
                views.append((detection.score, image, detection))
                views.sort(key=lambda v: v[0], reverse=True)
                del views[self.views_to_read:]
                if detection.score > event['score']:
                    event.update(score=detection.score, image=image, detection=detection, pose=self.pose)
            elif event is not None and now - event['last_seen'] >= self.sign_lost_time:
                self.save_board(name, self.events.pop(name))

    def save_board(self, camera, event, same_board_distance=1.0):
        """Read a board's views (vote), save its best view, submit the clue."""
        x, y = event['pose']
        image, detection = event['image'], event['detection']
        votes, frames = {}, 0
        largest_letters = max(letter_height(d) for _, _, d in event['views'])
        if self.reader is not None:
            for _, view, view_detection in event['views']:
                if letter_height(view_detection) < self.min_letter_px:
                    continue  # too far away to read - it would only add a wrong guess
                reading = self.reader.read(sign_detector.flatten(view, view_detection))
                # Only plausible readings vote: the clue type must be within one
                # letter of a real keyword (a blurry view reads as garbage)
                if reading['value'] and reading['type_distance'] <= 1:
                    key = (reading['type'], reading['value'])
                    votes[key] = votes.get(key, 0.0) + view_detection.area
                    frames += 1
        reading = max(votes, key=votes.get) if votes else None
        previous = next((b for b in self.boards
                         if math.hypot(b['pose'][0] - x, b['pose'][1] - y) < same_board_distance
                         or (reading is not None and b['reading'] is not None
                             and b['reading'][0] == reading[0])), None)
        if previous is not None:
            for key, weight in previous['votes'].items():
                votes[key] = votes.get(key, 0.0) + weight
            frames += previous['frames']
            reading = max(votes, key=votes.get) if votes else None
            self.boards.remove(previous)
        better_view = previous is None or event['score'] > previous['score']
        number = len(self.boards) + 1 if previous is None else previous['number']

        os.makedirs(self.folder, exist_ok=True)
        if better_view:
            if previous is not None:
                for suffix in ('.png', '_banner.png'):
                    try:
                        os.remove(os.path.join(self.folder, previous['stem'] + suffix))
                    except OSError:
                        pass
            stem = f'board_{number:02d}_{camera}_x{x:+.2f}_y{y:+.2f}'
            cv2.imwrite(os.path.join(self.folder, stem + '.png'), image)
            cv2.imwrite(os.path.join(self.folder, stem + '_banner.png'),
                        sign_detector.flatten(image, detection))
        else:
            stem = previous['stem']
        self.boards.append({'number': number, 'pose': (x, y) if better_view else previous['pose'],
                            'score': event['score'] if better_view else previous['score'],
                            'stem': stem, 'votes': votes, 'frames': frames, 'reading': reading})

        if reading is None:
            text = (f'not read (letters ~{largest_letters:.0f} px, need {self.min_letter_px:.0f})'
                    if largest_letters < self.min_letter_px else 'not read')
        else:
            agree = votes[reading] / max(1e-9, sum(votes.values()))
            text = f'reads {reading[0]} = {reading[1]} ({agree:.0%} of the vote over {frames} views)'
        what = 'captured' if previous is None else 'seen again'
        self.get_logger().info(
            f'Clue board {number} {what} by the {camera} camera '
            f'({detection.area_fraction * 100:.1f}% of the image): {text} '
            f'-> {os.path.join(self.folder, stem)}.png')
        self.write_table()
        if reading is not None:
            msg = String()
            msg.data = f'{camera},{clue_reader.KEYWORDS.index(reading[0]) + 1},{reading[0]},{reading[1]}'
            self.identified_pub.publish(msg)
        if self.submit_clues and reading is not None:
            location = clue_reader.KEYWORDS.index(reading[0]) + 1
            if self.submitted.get(location) != reading[1]:
                self.submitted[location] = reading[1]
                msg = String()
                msg.data = f'{self.team_id},{self.team_password},{location},{reading[1]}'
                self.score_pub.publish(msg)
                self.get_logger().info(f'Sent to /score_tracker: {msg.data}')

    def write_table(self):
        with open(os.path.join(self.folder, 'clues_read.csv'), 'w') as table:
            table.write('board,clue type,location,value,vote share,views,image\n')
            for board in sorted(self.boards, key=lambda b: b['number']):
                if board['reading'] is None:
                    table.write(f"{board['number']},,,,,{board['frames']},{board['stem']}.png\n")
                    continue
                clue_type, value = board['reading']
                share = board['votes'][board['reading']] / max(1e-9, sum(board['votes'].values()))
                table.write(f"{board['number']},{clue_type},{clue_reader.KEYWORDS.index(clue_type) + 1},"
                            f"{value},{share:.2f},{board['frames']},{board['stem']}.png\n")

    def build_window(self, images):
        tiles = []
        for name, _ in CAMERAS:
            image = images.get(name)
            if image is None:
                tiles.append(np.zeros((240, 320, 3), np.uint8))
                continue
            image = image.copy()
            detection = self.latest_detections.get(name)
            if detection is not None:
                sign_detector.draw(image, detection, 'clue board')
            image = cv2.resize(image, (320, 240))
            cv2.putText(image, name, (8, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(image)
        mosaic = np.hstack(tiles)
        read = [f"{b['reading'][0]}={b['reading'][1]}" for b in sorted(self.boards, key=lambda b: b['number'])
                if b['reading'] is not None]
        lines = [self.status or (self.state or ''), 'clues: ' + ('  '.join(read) if read else '-')]
        for row, text in enumerate(lines):
            y = 20 + 20 * row
            cv2.putText(mosaic, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(mosaic, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return mosaic


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ClueFinder()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if node.show_cameras:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    # One thread: ROS callbacks, the camera work and the window all run here
    next_step = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if time.monotonic() >= next_step:
                next_step = time.monotonic() + node.period
                frame = node.step()
                if frame is not None:
                    cv2.imshow(WINDOW, frame)
            if node.show_cameras:
                cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        if node.events:
            node.flush()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
