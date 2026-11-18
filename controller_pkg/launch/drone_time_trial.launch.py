from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='controller_pkg',
            executable='drone_time_trial',
            output='screen',
            parameters=[{
                'team_id': 'TeamFizz',
                'team_password': 'pass1234',
            }],
        ),
    ])
