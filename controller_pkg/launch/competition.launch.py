"""Competition run: both drones with one command.

Start the competition simulation first (it spawns the first drone and the
score tracker), then:

    ros2 launch controller_pkg competition.launch.py

  1. spawns the second drone (drone2) in the free respawn box, its bridge,
     its flight controller and its clue_finder (drone2.launch.py). It waits on
     the ground for the first drone
  2. after drone1_delay seconds (default 5 - drone 2 has spawned and is
     listening by then), starts the first drone (drone_wall_ride.launch.py): it
     starts the timer and rides the walls (clues 5 - 8)
  3. drone 2 takes off once drone 1 has ridden the south wall for 6 s, reads
     clues 1 - 4 and lands. Drone 1 stops the timer at the end of its route.

Arguments: drone1_delay:=<s>, flight_params:=<yaml> (drone 2's parameter
overrides, see drone2.launch.py).

Restart the simulation before running it again: drone2 stays in the world
after this launch stops, and spawning it a second time fails.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_dir = Path(get_package_share_directory('controller_pkg')) / 'launch'
    drone2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource((launch_dir / 'drone2.launch.py').as_posix()),
        launch_arguments={'flight_params': LaunchConfiguration('flight_params')}.items())
    drone1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource((launch_dir / 'drone_wall_ride.launch.py').as_posix()))
    return LaunchDescription([
        DeclareLaunchArgument('drone1_delay', default_value='5.0',
                              description='s after drone 2 spawns before drone 1 starts (and the timer)'),
        DeclareLaunchArgument('flight_params', default_value='',
                              description="optional YAML of drone 2's parameter overrides"),
        drone2,
        TimerAction(period=LaunchConfiguration('drone1_delay'), actions=[drone1]),
    ])
