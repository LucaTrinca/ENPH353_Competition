#!/usr/bin/env python3
"""Keyboard teleop for the drone.

Opens a window showing the front camera and IMU readings. While the window
is focused, hold:
    W / S       forward / backward force
    A / D       left / right force
    Space       up force
    Ctrl        down force
    Q / E       rotate left (CCW) / right (CW)
Press:
    L           save the current camera image (PNG) to <save_dir>

Publishes force commands on /cmd_vel (geometry_msgs/Twist, forces in N):
    linear.x = forward, linear.y = left, linear.z = up
    angular.z = yaw rate [rad/s]
The drone plugin compensates gravity once the drone has taken off, so with no
keys pressed the drone hovers (and drifts with the wind).
"""
import os
import tkinter as tk
from datetime import datetime

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from PIL import Image as PILImage
from PIL import ImageTk
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu


class DroneTeleop(Node):

    def __init__(self):
        super().__init__('drone_teleop')

        self.declare_parameter('horizontal_force', 1.0)  # N
        self.declare_parameter('vertical_force', 1.5)    # N
        self.horizontal_force = float(self.get_parameter('horizontal_force').value)
        self.vertical_force = float(self.get_parameter('vertical_force').value)
        self.declare_parameter('yaw_rate', 1.0)          # rad/s
        self.yaw_rate = float(self.get_parameter('yaw_rate').value)
        self.declare_parameter('save_dir', '~/drone_images')
        self.save_dir = os.path.expanduser(self.get_parameter('save_dir').value)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 1)
        self.create_subscription(Imu, '/imu', self.imu_callback, 1)

        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_imu = None

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def imu_callback(self, msg):
        self.latest_imu = msg

    def publish_force(self, forward, left, up, yaw=0.0):
        twist = Twist()
        twist.linear.x = forward * self.horizontal_force
        twist.linear.y = left * self.horizontal_force
        twist.linear.z = up * self.vertical_force
        twist.angular.z = yaw * self.yaw_rate
        self.cmd_pub.publish(twist)
        return twist

    def save_frame(self):
        """Save the latest camera frame as a PNG; returns the path or None."""
        if self.latest_frame is None:
            return None
        os.makedirs(self.save_dir, exist_ok=True)
        filename = datetime.now().strftime('frame_%Y%m%d_%H%M%S_%f.png')
        path = os.path.join(self.save_dir, filename)
        PILImage.fromarray(self.latest_frame).save(path)
        self.get_logger().info(f'Saved camera image: {path}')
        return path


class TeleopWindow:

    # X11 auto-repeat sends release+press pairs while a key is held; a release
    # is only accepted if no press follows within this delay
    RELEASE_DELAY_MS = 40
    LOOP_MS = 50  # 20 Hz

    def __init__(self, node):
        self.node = node
        self.pressed = set()
        self.pending_release = {}

        self.root = tk.Tk()
        self.root.title('Drone Teleop')
        self.image_label = tk.Label(self.root, text='Waiting for /camera/image_raw...',
                                    width=90, height=30)
        self.image_label.pack()
        self.info = tk.Label(self.root, font=('Monospace', 11), justify=tk.LEFT, anchor='w')
        self.info.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(self.root, text='W/S forward/back   A/D left/right   '
                                 'Space up   Ctrl down   Q/E rotate   L save image   '
                                 '(keep this window focused)'
                 ).pack(pady=(0, 6))
        self.save_status = 'No images saved yet'

        self.root.bind('<KeyPress>', self.on_press)
        self.root.bind('<KeyRelease>', self.on_release)
        self.root.bind('<FocusOut>', lambda _e: self.pressed.clear())
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(self.LOOP_MS, self.loop)
        self.photo = None

    @staticmethod
    def key_name(event):
        key = event.keysym.lower()
        if key.startswith('control'):
            return 'ctrl'
        return key

    def on_press(self, event):
        key = self.key_name(event)
        pending = self.pending_release.pop(key, None)
        if pending is not None:
            self.root.after_cancel(pending)
        # Only on the first press, not on auto-repeat while the key is held
        if key == 'l' and key not in self.pressed:
            path = self.node.save_frame()
            self.save_status = f'Saved {path}' if path else 'No camera image to save yet'
        self.pressed.add(key)

    def on_release(self, event):
        key = self.key_name(event)
        self.pending_release[key] = self.root.after(
            self.RELEASE_DELAY_MS, lambda: self.release(key))

    def release(self, key):
        self.pending_release.pop(key, None)
        self.pressed.discard(key)

    def axis(self, positive, negative):
        return float(positive in self.pressed) - float(negative in self.pressed)

    # Max ROS callbacks handled per loop. The IMU alone publishes at 100 Hz, so a
    # single spin_once per loop would starve the camera callback.
    MAX_CALLBACKS_PER_LOOP = 50

    def loop(self):
        for _ in range(self.MAX_CALLBACKS_PER_LOOP):
            rclpy.spin_once(self.node, timeout_sec=0.0)

        twist = self.node.publish_force(forward=self.axis('w', 's'),
                                        left=self.axis('a', 'd'),
                                        up=self.axis('space', 'ctrl'),
                                        yaw=self.axis('q', 'e'))

        if self.node.latest_frame is not None:
            self.photo = ImageTk.PhotoImage(PILImage.fromarray(self.node.latest_frame))
            self.image_label.configure(image=self.photo, width=0, height=0)

        lines = [f'Force cmd [N]   fwd {twist.linear.x:+5.1f}   left {twist.linear.y:+5.1f}   '
                 f'up {twist.linear.z:+5.1f}   yaw rate {twist.angular.z:+4.1f} rad/s']
        imu = self.node.latest_imu
        if imu is not None:
            a, w = imu.linear_acceleration, imu.angular_velocity
            lines.append(f'IMU accel [m/s^2]  x {a.x:+6.2f}  y {a.y:+6.2f}  z {a.z:+6.2f}')
            lines.append(f'IMU gyro  [rad/s]  x {w.x:+6.2f}  y {w.y:+6.2f}  z {w.z:+6.2f}')
        else:
            lines.append('IMU: waiting for /imu...')
        lines.append(self.save_status)
        self.info.configure(text='\n'.join(lines))

        self.root.after(self.LOOP_MS, self.loop)

    def close(self):
        self.pressed.clear()
        self.node.publish_force(0.0, 0.0, 0.0)
        self.root.destroy()

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.close()


def main(args=None):
    rclpy.init(args=args)
    node = DroneTeleop()
    TeleopWindow(node).run()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
