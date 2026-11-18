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
            executable='drone_road_follower',
            output='screen',
            parameters=[{
                'team_id': 'TeamFizz',
                'team_password': 'pass1234',
                'cruise_altitude': 0.3,
                'max_search_altitude': 0.6,
                'follow_duration': 60.0,
                'cruise_speed': 0.4,
                'down_camera_fov_deg': 134.0,
                'altitude_kp': 3.0,
                'altitude_ki': 0.5,
                'altitude_kd': 2.0,
                'lateral_kp': 10.0,
                'lateral_ki': 4.0,
                'lateral_kd': 4.0,
                'speed_kp': 4.0,
                'speed_ki': 3.0,
                'yaw_kp': 1.5,
                'max_horizontal_force': 4.0,
                'wind_feedforward': True,
                'wind_ff_gain': 1.0,
                'wind_filter_tau': 0.3,
                'drone_mass': 1.03,
                'use_range_sensor': True,
                'air_drag_coeff': 1.0,
                'max_total_horizontal_force': 8.0,
                'show_debug': True,
                'show_front_camera': True,
            }],
        ),
    ])
