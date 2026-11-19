from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    bridge_config = (Path(get_package_share_directory('controller_pkg'))
                     / 'config' / 'drone_sensors_bridge.yaml')

    return LaunchDescription([
        # Down camera, depth range sensor, altimeter and IMU (not in the competition bridge)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            parameters=[{'config_file': bridge_config.as_posix()}],
            output='screen',
        ),
        Node(
            package='controller_pkg',
            executable='drone_sign_tour',
            output='screen',
            # Gains are accelerations (m/s^2), scaled by drone_mass into forces
            parameters=[{
                'team_id': 'TeamFizz',
                'team_password': 'pass1234',
                'drone_mass': 20.0,
                'cruise_speed': 0.4,
                'view_distance': 0.35,
                'transit_altitude': 0.4,
                'view_altitude': 0.4,
                'mountain_altitude': 2.3,
                'dwell': 2.0,
                'photo_dir': '~/drone_images',
                'show_front_camera': True,
            }],
        ),
    ])
