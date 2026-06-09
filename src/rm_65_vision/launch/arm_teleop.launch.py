from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rm_65_task_share = get_package_share_directory('rm_65_task')
    config_file = os.path.join(rm_65_task_share, 'config', 'arm_teleop_mapping.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('camera_id', default_value='0'),
        DeclareLaunchArgument('tracking_side', default_value='right'),
        Node(
            package='rm_65_vision',
            executable='arm_teleop_tracker',
            name='arm_teleop_tracker',
            output='screen',
            parameters=[
                {'camera_id': LaunchConfiguration('camera_id')},
                {'tracking_side': LaunchConfiguration('tracking_side')},
                {'show_debug': True},
                {'config_file': config_file},
                {'mirror_preview': True},
            ],
        ),
    ])
