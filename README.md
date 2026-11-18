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

# Drone keyboard teleop (keep the teleop window focused)
#   Space/Ctrl up/down, W/S forward/back, A/D left/right, Q/E rotate, L save camera image,
#   R reset the drone to its spawn pose
ros2 launch controller_pkg drone_teleop.launch.py
```

## Drone sensors

| Sensor | Topic | Use |
| :--- | :--- | :--- |
| Down camera, 320x240 at 15 Hz, 134 deg FOV | `/down_camera/image_raw` | Road following, ground velocity (optical flow) |
| Front camera, 720x480 at 5 Hz | `/camera/image_raw` | Clue boards (only displayed for now) |
| Altimeter, 30 Hz | `/altimeter` | Altitude hold (height relative to the spawn height) |
| IMU, 100 Hz | `/imu` | Virtual wind sensor (road follower) |

**Wind sensor.** Gazebo has no anemometer, so the road follower estimates the wind from the
IMU: wind force = mass x IMU acceleration - commanded force, and wind velocity = wind force /
1 N per m/s + optical-flow ground velocity. It cancels that wind force in real time
(feedforward, `wind_feedforward` / `wind_ff_gain`) and publishes the estimate on `/drone/wind`
(geometry_msgs/Vector3Stamped, m/s, x forward, y left).

The competition launch only bridges `/camera/image_raw`. The drone launch files bridge the
other sensors with `controller_pkg/config/drone_sensors_bridge.yaml`.

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
- Don't put `: ` in URDF/xacro comments. The competition launch file parses the robot
  description as YAML and fails to launch.
