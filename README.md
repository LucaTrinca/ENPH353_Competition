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

# Drone keyboard teleop (keep the teleop window focused)
#   Space/Ctrl up/down, W/S forward/back, A/D left/right, Q/E rotate, L save camera image
ros2 launch controller_pkg drone_teleop.launch.py
```

## Notes

- Wind is disabled on the drone chassis for now (`my_robot_pkg/urdf/drone_gazebo.xacro`).
  The rules require `enable_wind` true for flying robots, so re-enable it before the competition.
- Don't put `: ` in URDF/xacro comments. The competition launch file parses the robot
  description as YAML and fails to launch.
