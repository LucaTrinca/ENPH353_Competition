from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='controller_pkg',
            executable='time_trial_controller',
            output='screen',
            parameters=[{
                'team_id': 'TeamFizz',
                'team_password': 'pass1234',
                'linear_speed': 0.4,
                'drive_duration': 30.0,
                'kp': 2.0,
                'ki': 0.0,
                'kd': 0.15,
                'max_angular': 1.5,
                'show_camera': True,
            }],
        ),
    ])
