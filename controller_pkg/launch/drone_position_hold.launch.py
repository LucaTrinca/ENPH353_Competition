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
            executable='drone_position_hold',
            output='screen',
            parameters=[{
                'hold_altitude': 0.3,
                'kp': 12.0,
                'ki': 3.0,
                'kd': 5.0,
                'wind_feedforward': True,
                'wind_ff_gain': 1.0,
            }],
        ),
    ])
