from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    bridge_config = (Path(get_package_share_directory('controller_pkg'))
                     / 'config' / 'drone_sensors_bridge.yaml')

    return LaunchDescription([
        # Down camera, side cameras, range sensors, altimeter and IMU (not in the competition bridge)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            parameters=[{'config_file': bridge_config.as_posix()}],
            output='screen',
        ),
        # Flight: rides the walls, starts / stops the timer
        Node(
            package='controller_pkg',
            executable='drone_wall_ride',
            output='screen',
            # Gains are accelerations (m/s^2), scaled by drone_mass into forces
            parameters=[{
                'team_id': 'TeamFizz',
                'team_password': 'pass1234',
                'drone_mass': 20.0,
                'ride_altitude': 0.35,
                'wall_distance': 0.4,
                'corner_distance': 0.4,
                'cruise_speed': 0.35,
            }],
        ),
        # Cameras: finds, reads and submits the clue boards, shows the camera
        # window. A separate process so it can never slow the flight control.
        Node(
            package='controller_pkg',
            executable='clue_finder',
            output='screen',
            parameters=[{
                'team_id': 'TeamFizz',
                'team_password': 'pass1234',
                'show_cameras': True,
                # '' = trained CNN (models/clue_chars_cnn.onnx); a .yml = older SVM
                'clue_model': '',
                # Send readings to /score_tracker (location = clue type number, SIZE 1 ... BANDIT 8)
                'submit_clues': True,
                # Seconds between saved raw camera frames (0 = off)
                'record_interval': 0.0,
                'photo_dir': '~/drone_images',
            }],
        ),
    ])
