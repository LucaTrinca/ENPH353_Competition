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
# hover. Submits clues 5 - 8 (~2 min; clues 1 - 4 are the second drone's). tunnel_and_clue8:=false
# stops at the north-east corner instead. Starts two programs: drone_wall_ride (flight only) and
# clue_finder (cameras: finds, reads and submits the clue boards to the score
# tracker, saves them to ~/drone_images/clues_<time>/ with clues_read.csv, and
# opens the camera window - front and left; the right camera is off). They are separate processes so
# the camera work can't slow the flight control.
ros2 launch controller_pkg drone_wall_ride.launch.py

# Competition run - both drones with one command (after the simulation is up):
# spawns drone 2, starts drone 1 5 s later (it starts the timer), drone 2 takes
# off once drone 1 is clear of the spawn box. Restart the simulation between
# runs (drone2 stays in the world). drone1_delay:=<s>, flight_params:=<yaml>.
ros2 launch controller_pkg competition.launch.py

# Second drone (clues 1 - 4): spawns "drone2" in the free respawn box (north-west
# corner), waits until the first drone has left the box, reads clue 1 from just
# south of it, rides the west wall north to clue 2, on to the north-west corner,
# east along the north wall and down to clue 3, then south of the road loop
# to clue 4 (read with the right camera), and lands. ~75 s. Everything
# under /drone2; it never touches the timer and only submits clues 1 - 4 (the
# first drone only 5 - 8). Start it after the simulation, alongside
# drone_wall_ride.launch.py. flight_params:=<yaml> overrides its parameters.
ros2 launch controller_pkg drone2.launch.py

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

## Second drone

The rules allow two custom robots. `drone2.launch.py` spawns the same drone
model with three xacro arguments (`my_robot.urdf.xacro`):

| Argument | First drone | Second drone | Why |
|---|---|---|---|
| `prefix` | `` | `drone2/` | its sensor topics become `/drone2/imu`, `/drone2/camera/image_raw`, ... |
| `left_range` | `false` | `true` | a left range strip (`/drone2/left_depth`) for riding the west wall on its left |
| `sensor_systems` | `true` | `false` | the IMU and altimeter *systems* are world-wide: one copy serves every drone. A second copy published every altimeter twice with different reference heights (0.1 m altitude jumps; once sent drone 2 to 14 m) |

Its controller is `drone_west_ride` (a subclass of `drone_wall_ride`: same wind
cancelling, optical flow, altitude hold and stuck watchdog) and its own
`clue_finder` (front camera, plus the right camera only at clue 4; submits
locations 1 - 4). Topic names in both nodes are relative, so the first drone
runs at the root and the second under `drone2`; `/clock` and `/score_tracker`
are shared.

Route (x east, y north) and why:

| Step | Where | How it reads / navigates |
|---|---|---|
| Clue 1 (-5.81, -1.64), faces south | hover 0.66 m south of it | front camera |
| Clue 2 (-5.16, 1.11 - 1.6), faces south | ride the west wall north at 0.5 m, stop when seen | front camera, ~25 deg from the side: the fir at (-5, 0.9) stands right in front of it |
| Corner | on up the west wall to 0.5 m from the north wall | turn east here: both walls are in range before and after |
| Clue 3 (-4.0, 1.67), faces west | east along the north wall (north of the fir at (-5, 1.8)), then down to 0.65 m in front of it | front camera, head on |
| Clue 4 (-0.83, 0 - 0.63), faces north | diagonally to the south-west of the road loop, east along a lane just south of it (y 0.05: the truck drives the loop, and under the down camera it drags the optical flow along) to 0.5 m from the plateau, turn to -45 deg, slide north | right camera (looking south-west, 45 deg off the board's facing); it slides until the board is centred in the image. Straight north of the board an oak hides it |
| End | | lands, out of drone 1's way |

Things the route is built around:

- **The west half has almost no texture for the optical flow**: flat green paint
  and asphalt with straight white edges. Over plain paint the flow measures
  nothing; along a lone straight edge it only sees the motion across it, and
  phase correlation reported ~0 along it - the estimate then said "standing
  still" while the drone sped up (1.7 m of error in one test). Drone 2 uses a
  wider crop of the down image (`flow_crop` 0.55) and `flow_min_isotropy` 0.1:
  when the texture's gradients all run one way, only the velocity across the
  lines is corrected and the IMU carries the rest.
- **So it navigates by the range sensors**, and dead reckoning only bridges
  gaps: facing north on the west wall line, x from the west wall (left range)
  and y from the north wall (front range); facing east near the north wall, y
  from the north wall (left range); facing within 50 deg of east on the north
  wall line or south of the loop, x from the plateau's west face (front range;
  the face is a plane at x 0.14 at the sensor's height). A straight line is fitted
  through the last readings over time, so a moving drone gets position *and*
  velocity fixes, and a tree passing through the beam is ignored.
- A row of fir trees stands at x = -5.0. Their crowns (0.51 m tall, 0.17 m
  radius, ~5 cm collision) confuse the flow and the range sensors, so the drone
  rides 0.5 m from the west wall and goes round the north side of the fir next
  to clue 3 instead of through the gap (clue 2 stands in that gap).
- The truck drives round the road loop (0.08 m tall) and the pedestrian crosses
  at y -0.5: drone 2 flies over the truck at 0.35 m and never near the crossing.
- A gust at liftoff can push a drone ~0.6 m, so drone 2 only takes off once
  drone 1 has ridden the south wall for 6 s (or after 5 s if drone 1 isn't
  running).
- `clues_3_4:=false` (in a `flight_params` file) stops after clue 2.

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
joined), crops each like the notebook (ink box + 4 px, 32x32) and classifies it with a CNN.
The clue type is snapped to the closest of the 8 keywords. From the drone, letters need to be
~8 px or taller in the camera image (boards up to ~2 m away with the 1280x960 side cameras).

Values can be 12 characters long (`FAME N GLORY`; the generator's default set has
`D NEGATIVITY`), and the 12th is half cut off by the banner's own edge - the score tracker
still wants it. So the banner crop stays 2 px inside the blue frame (it was 6 px, which cut
off the half letter), the reader allows 12 value characters, and strips of the frame along the
banner's edge (taller than the letters, or reaching the banner's top / bottom) are not
characters. A half-cut round letter (O, Q, C, G, 6, 0) is still often misread: its left sliver
looks the same.

The CNN (`controller_pkg/models/clue_chars_cnn.onnx`) is the team notebook's architecture
(Conv32-Pool-Conv64-Pool-Conv64-Dense128-Dropout 0.3, softmax over 0-9 A-Z). Its input is the
crop's "blueness" (B - (G+R)/2) instead of grey, which ignores the lighting. It is trained on
~27,000 characters from banners drawn exactly as the competition does (clue_banner.png,
UbuntuMono-R size 90, values up to 12 characters) and degraded like the drone camera
(lighting, crop errors, 5-40 px letters, noise), each cut by the same segmentation the reader
uses (98.3% validation character accuracy).

Whole clues read right (type and value, spaces ignored - how the score tracker checks), against
the team Colab's pipeline (inner-panel unwarp, contour slicing, grey 32x32 CNN trained on its
100 synthetic boards):

| Test | This reader | Colab pipeline |
|---|---|---|
| team photos, 121 sim frames of the 8 competition clues (8 have no complete board) | 113 (93%), 0 wrong | 79 (65%) |
| 160 views of 40 boards with random text (up to 12 chars) spawned in the sim, drone camera settings, 0.45 - 1.1 m, up to 35 deg | 119 (74%) | 30 (19%) |
| boards saved in our sim runs (236) | 226 | - |

Most of the remaining misses on random boards are the half-cut 12th character, or views too
far / oblique to find the board.

Votes across views are weighted by board size times the CNN's confidence, so a small, unsure
view counts for little. The drone runs the model with OpenCV's dnn module; training needs
PyTorch and onnx (CPU wheels are enough, ~8 minutes on a laptop):

```bash
python3 -m venv --system-site-packages ~/clue_venv
~/clue_venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
~/clue_venv/bin/pip install "numpy<2" "ml-dtypes<0.5" "onnx<1.18"   # numpy 2 breaks ROS's OpenCV
cd ~/2026_competition_ws/src/controller_pkg && ~/clue_venv/bin/python -m controller_pkg.train_clue_reader
cd ~/2026_competition_ws && colcon build --packages-select controller_pkg
```

`--model svm` trains the old SVM instead (no PyTorch needed); pass a model's path as
`clue_finder`'s `clue_model` parameter to use it.

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
