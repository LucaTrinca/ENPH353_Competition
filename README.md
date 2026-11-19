# ENPH 353 2026 Fizz Detective - team packages

This repository is the `src/` folder of `~/2026_competition_ws`.

| Package | Description |
| :--- | :--- |
| `controller_pkg` | Our ROS2 nodes: drone time trial, drone keyboard teleop, car time trial / PID road following. |
| `drone_plugins` | Gazebo system plugin (`DroneForceController`) that flies the drone with force commands from `/cmd_vel`. |
| `my_robot_pkg` | Robot model (modified from [ENPH353/my_robot_pkg](https://github.com/ENPH353/my_robot_pkg)): drone (default) or car via the `robot_type` xacro arg. |

The competition world is not included; clone it next to these packages.

## Setup

```bash
mkdir -p ~/2026_competition_ws
git clone <this repo url> ~/2026_competition_ws/src
cd ~/2026_competition_ws/src
git checkout time-trials
git clone https://github.com/ENPH353/2026_competition.git

cd ~/2026_competition_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

## Running

Open a new terminal for each command and source the workspace first: `source ~/2026_competition_ws/install/setup.bash`.

```bash
# Simulation (spawns the drone)
ros2 launch my_robot_bringup my_robot.launch.py

# Drone time trial: start timer, take off, fly forward, hover, stop timer
ros2 launch controller_pkg drone_time_trial.launch.py

# Drone road following: start timer, climb to 0.3 m, follow the road with the
# downward camera for 60 s (rightmost branch at junctions), stop timer, keep
# hovering. Opens the down camera debug window and a front camera window.
# Start it with the drone on the ground (it zeroes the altimeter there).
ros2 launch controller_pkg drone_road_follow.launch.py

# Drone wall ride: start timer, take off, ride the south wall (behind the spawn)
# to the east corner (stepping out on the way to read clue 6, which faces west
# 0.64 m in from the wall), turn, ride the east wall to the north-east corner,
# turn, ride the north wall to the tunnel and touch it (+5 bonus), then climb
# to 2.3 m, cross onto the mountain's plateau and read clue 8, stop timer,
# hover. Reads clues 1, 5, 6, 7 and 8 (~2.5 min). tunnel_and_clue8:=false
# stops at the north-east corner instead. Starts two programs: drone_wall_ride (flight only) and
# clue_finder (cameras: finds, reads and submits the clue boards to the score
# tracker, saves them to ~/drone_images/clues_<time>/ with clues_read.csv, and
# opens the left | front | right camera window). They are separate processes so
# the camera work can't slow the flight control.
ros2 launch controller_pkg drone_wall_ride.launch.py

# Drone keyboard teleop (keep the teleop window focused)
#   Space/Ctrl up/down, W/S forward/back, A/D left/right, Q/E rotate, L save camera image,
#   R reset the drone to its spawn pose
ros2 launch controller_pkg drone_teleop.launch.py
```

## Drone sensors

| Sensor | Topic | Use |
| :--- | :--- | :--- |
| Down camera, 320x240 at 15 Hz, 134 deg FOV | `/down_camera/image_raw` | Road following, ground velocity (optical flow) |
| Front camera, 720x480 at 5 Hz, tilted down 0.6 rad | `/camera/image_raw` | Clue boards |
| Left / right cameras, 1280x960 at 5 Hz, 70 deg FOV, tilted down 0.25 rad | `/left_camera/image_raw`, `/right_camera/image_raw` | Clue boards beside the drone (wall ride) |
| Right range sensor (32x4 depth strip, 1.2 rad wide), 30 Hz | `/right_depth` | Distance to the wall on the right, and the wall's angle (yaw correction) |
| Front range sensor (8x4 depth strip), 15 Hz | `/front_depth` | Wall ahead (corners) |
| Down range sensor (8x8 depth camera), 30 Hz | `/down_depth` | Height above the ground (follows terrain) |
| Altimeter, 30 Hz | `/altimeter` | Fallback height when the range sensor has no reading (relative to the spawn height) |
| IMU, 100 Hz | `/imu` | Velocity / position dead reckoning, wind (disturbance) estimate |

**Mass.** The drone is 20 kg, the maximum allowed. All controller gains and limits are
accelerations (m/s^2) multiplied by `drone_mass`, and the plugin's limit is `max_accel` (m/s^2) -
set it to whatever acceleration limit the competition enforces.

**Wind.** The world's wind plugin (gz WindEffects) applies a force *proportional to mass*
(scaling factor 1: force = mass x (wind velocity - drone velocity) per second), so the heavy
drone does **not** escape it: the wind (1 m/s, direction swinging +/-137 deg every 10 s, +/-80%
gusts) pushes with up to ~1.8 m/s^2. The wall ride measures that push with a disturbance
observer (IMU acceleration minus commanded acceleration, low-passed) and cancels it. The road
follower's older wind estimator assumed a mass-independent force, which is why it read
14-220 m/s at 20 kg - its feedforward is off.

The competition launch only bridges `/camera/image_raw`. The drone launch files bridge the
other sensors with `controller_pkg/config/drone_sensors_bridge.yaml`.

## Clue boards

`controller_pkg/sign_detector.py` finds a complete clue board in an image. It is the team
notebook's clue finder (blue mask, 4 corners, perspective warp) with these changes: every blue
region is tried (not only the largest - the lake is blue too), the board must not touch the
image border, it must be a clean quadrilateral with a solid blue frame on all four sides and a
light banner inside, slivers of background touching it are removed, a finer mask is tried for
distant boards (thin frame), and the crop is the banner itself rather than a fixed border. On
the team's 121 photos it finds 113 boards (the rest are cut off or hidden) with no false
detections in ~1100 sign-free frames.

`controller_pkg/clue_reader.py` reads a banner: it cuts the two lines into characters on the
monospaced font's grid (spaces are empty grid slots; merged or broken letters are split or
joined), crops each like the notebook (ink box + 4 px, 32x32) and classifies it with HOG
features + an SVM. The clue type is snapped to the closest of the 8 keywords. It reads all 113
detected boards in the team's photos correctly; from the drone, letters need to be ~8 px or
taller in the camera image (boards up to ~2 m away with the 1280x960 side cameras).

The SVM (`controller_pkg/models/clue_chars_svm.yml`) is trained on banners drawn exactly as the
competition does (clue_banner.png, UbuntuMono-R size 90) and degraded like the drone camera
(lighting, crop errors, 5-40 px letters, noise):

```bash
cd ~/2026_competition_ws/src/controller_pkg && python3 -m controller_pkg.train_clue_reader --banners 3000
cd ~/2026_competition_ws && colcon build --packages-select controller_pkg
```

To use the notebook's CNN instead, export it to ONNX in Colab and pass its path as the wall
`clue_finder`'s `clue_model` parameter (OpenCV runs it; no TensorFlow needed on the robot):

```python
!pip install tf2onnx
import tensorflow as tf, tf2onnx
model = tf.keras.models.load_model('fizz_detective_cnn.h5')
spec = (tf.TensorSpec((None, 32, 32, 1), tf.float32, name='input'),)
tf2onnx.convert.from_keras(model, input_signature=spec, output_path='fizz_detective_cnn.onnx')
```

`clue_finder` sends each reading to `/score_tracker` as soon as the board has left the view
(`submit_clues`, on by default) with location = the clue type's number (SIZE 1, VICTIM 2,
CRIME 3, TIME 4, PLACE 5, MOTIVE 6, WEAPON 7, BANDIT 8 - the row order of the competition's
clues.csv). Only readings whose clue type is within one letter of a keyword count, and a clue
is sent again only if a later view changes the answer (the score tracker keeps the latest).
Boards still in view at the final corner are submitted before the timer is stopped.

## Resetting the drone

The drone plugin teleports the drone back to its spawn pose (on the ground, stopped and
disarmed) when it gets a message on its reset topic. Use any of these:

- **R** in the teleop window
- `ros2 topic pub --once /drone/reset std_msgs/msg/Empty` (needs one of the drone launch
  files running, since they bridge `/drone/reset`)
- `gz topic -t /model/my_robot/reset -m gz.msgs.Empty -p ''` (works without the bridge)

Pressing R in the Gazebo window (the competition's ResetSimPlugin) does **not** reset the
drone. That plugin snapshots spawn poses before the drone has spawned, so it only resets the
NPCs.

## Notes

- Wind is enabled on the drone chassis, as the rules require. The drone plugin's own
  `linear_drag` is 0 because the wind plugin already applies air drag; set it to 1.0 if you
  ever turn wind off, or the drone will never slow down.
- Gazebo transport uses multicast on your network interface. If Wi-Fi drops, sim topics can
  stop. `export GZ_IP=127.0.0.1` in every terminal keeps the sim on loopback.
- The down range sensor is a tiny depth camera, not a `gpu_lidar`: with the competition world's
  `ogre` render engine a gpu_lidar only ever reports its minimum range.
- Gazebo's `air_speed` sensor doesn't see the wind (it only uses the drone's own velocity), so it
  can't act as a wind sensor; the wind is estimated from the IMU instead.
- Don't put `: ` in URDF/xacro comments. The competition launch file parses the robot
  description as YAML and fails to launch.
