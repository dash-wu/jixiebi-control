from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rm_gazebo_share = get_package_share_directory('rm_gazebo')
    rm_65_config_share = get_package_share_directory('rm_65_config')
    rm_65_vision_share = get_package_share_directory('rm_65_vision')
    rm_65_task_share = get_package_share_directory('rm_65_task')

    mapping_file = os.path.join(rm_65_task_share, 'config', 'arm_teleop_mapping.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('camera_id', default_value='0'),
        DeclareLaunchArgument('tracking_side', default_value='right'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(rm_gazebo_share, 'launch', 'gazebo_65_gesture_demo.launch.py')
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(rm_65_config_share, 'launch', 'gazebo_moveit_demo.launch.py')
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(rm_65_vision_share, 'launch', 'arm_teleop.launch.py')
            ),
            launch_arguments={
                'camera_id': LaunchConfiguration('camera_id'),
                'tracking_side': LaunchConfiguration('tracking_side'),
            }.items(),
        ),
        Node(
            package='rm_65_task',
            executable='teleop_controller',
            name='teleop_controller',
            output='screen',
            parameters=[
                {'use_sim': True},
                {'use_sim_time': True},
                {'mapping_file': mapping_file},
            ],
        ),
    ])
