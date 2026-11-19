from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    bridge_config = (Path(get_package_share_directory('controller_pkg'))
                     / 'config' / 'drone_sensors_bridge.yaml')

    return LaunchDescription([
        # Down camera, altimeter and IMU (not in the competition bridge)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            parameters=[{'config_file': bridge_config.as_posix()}],
            output='screen',
        ),
        Node(
            package='controller_pkg',
            executable='drone_teleop',
            output='screen',
            parameters=[{
                'horizontal_accel': 1.0,
                'vertical_accel': 1.5,
                'drone_mass': 20.0,
                'yaw_rate': 1.0,
                'save_dir': '~/drone_images',
            }],
        ),
    ])
