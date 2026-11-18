from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # The competition bridge doesn't include the IMU, so bridge it here
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/imu@sensor_msgs/msg/Imu[gz.msgs.IMU'],
            output='screen',
        ),
        Node(
            package='controller_pkg',
            executable='drone_teleop',
            output='screen',
            parameters=[{
                'horizontal_force': 1.0,
                'vertical_force': 1.5,
                'yaw_rate': 1.0,
                'save_dir': '~/drone_images',
            }],
        ),
    ])
