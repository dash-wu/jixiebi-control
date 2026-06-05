from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('rm_65_vision')
    hand_eye = os.path.join(pkg_share, 'config', 'hand_eye.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('camera_id', default_value='0'),
        DeclareLaunchArgument('arm_image_topic', default_value='/arm_camera/image_raw'),
        Node(
            package='rm_65_vision',
            executable='gesture_recognizer',
            name='gesture_recognizer',
            output='screen',
            parameters=[{'camera_id': LaunchConfiguration('camera_id')}],
        ),
        Node(
            package='rm_65_vision',
            executable='grasp_aligner',
            name='grasp_aligner',
            output='screen',
            parameters=[
                {'image_topic': LaunchConfiguration('arm_image_topic')},
                {'hand_eye_config': hand_eye},
            ],
        ),
    ])
