"""Second drone: spawn it, bridge it and read clues 1 - 4 (west half of the arena).

Start the competition simulation first (it spawns the first drone), then:

    ros2 launch controller_pkg drone2.launch.py

Everything for this drone lives under the drone2 namespace. It spawns in the
free respawn box (bottom-left), 0.72 m north-west of the first drone, never
touches the competition timer and only submits clues 1 - 4.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

NAMESPACE = 'drone2'
# North-west corner of the free respawn box (x -6 .. -5, y -2.75 .. -1.75):
# 0.72 m from the first drone, 3 cm from the west wall, just south of clue 1
SPAWN = {'x': '-5.82', 'y': '-1.85', 'z': '0.1', 'Y': '1.57'}


def flight_controller(context):
    """drone_west_ride, plus the parameters in flight_params:=<yaml> if given."""
    extra = LaunchConfiguration('flight_params').perform(context)
    return [
        Node(
            package='controller_pkg',
            executable='drone_west_ride',
            namespace=NAMESPACE,
            output='screen',
            parameters=[{
                'drone_mass': 20.0,
                'ride_altitude': 0.35,
                'start_x': float(SPAWN['x']),
                'start_y': float(SPAWN['y']),
                'west_wall_distance': 0.5,
                # The west half is flat paint and straight road edges: a wider
                # view of the ground has more to track, and along a lone edge
                # only the motion across it is measured
                'flow_crop': 0.55,
                'flow_min_isotropy': 0.1,
            }] + ([extra] if extra else []),
        ),
    ]


def generate_launch_description():
    urdf = Path(get_package_share_directory('my_robot_pkg')) / 'urdf' / 'my_robot.urdf.xacro'
    bridge_config = Path(get_package_share_directory('controller_pkg')) / 'config' / 'drone2_bridge.yaml'
    description = ParameterValue(
        Command(['xacro ', urdf.as_posix(), f' prefix:={NAMESPACE}/', ' left_range:=true',
                 ' sensor_systems:=false']),
        value_type=str)

    return LaunchDescription([
        # Optional YAML of drone_west_ride parameters (tuning / tests), e.g.
        # flight_params:=my_params.yaml with  drone2/drone_wall_ride: {ros__parameters: {...}}
        DeclareLaunchArgument('flight_params', default_value=''),
        # Publishes /drone2/robot_description for the spawner
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=NAMESPACE,
            parameters=[{'robot_description': description}],
            output='screen',
        ),
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=['-topic', f'/{NAMESPACE}/robot_description', '-name', NAMESPACE,
                       '-x', SPAWN['x'], '-y', SPAWN['y'], '-z', SPAWN['z'], '-Y', SPAWN['Y']],
            output='screen',
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='drone2_bridge',
            parameters=[{'config_file': bridge_config.as_posix()}],
            output='screen',
        ),
        # Give the spawn a moment before the controllers start
        TimerAction(period=3.0, actions=[
            OpaqueFunction(function=flight_controller),
            Node(
                package='controller_pkg',
                executable='clue_finder',
                namespace=NAMESPACE,
                output='screen',
                parameters=[{
                    'team_id': 'TeamFizz',
                    'team_password': 'pass1234',
                    'show_cameras': True,
                    # Front camera for clues 1 - 3, right camera for clue 4 (only
                    # subscribed - so only rendered - while sliding past it)
                    'cameras': ['front', 'right'],
                    'side_camera_states': ['TURN4', 'SLIDE', 'CLUE4_HOLD'],
                    # This drone's clues (the first drone submits 5 - 8)
                    'submit_locations': [1, 2, 3, 4],
                    # Every flying state of drone_west_ride (each board is read from
                    # its viewing point, and seen on the way there)
                    'active_states': ['TAKEOFF', 'CLUE1', 'RIDE', 'FACE', 'CLUE_HOLD', 'TO_CORNER',
                                      'TURN_EAST', 'NORTH_WALL', 'TO_CLUE3', 'CLUE3', 'TO_LOOP', 'BOTTOM',
                                      'TURN4', 'SLIDE', 'CLUE4_HOLD', 'RECOVER'],
                    'photo_dir': '~/drone_images/drone2',
                }],
            ),
        ]),
    ])
