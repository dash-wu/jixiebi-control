import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import xacro


def generate_launch_description():
    package_name = 'rm_gazebo'
    robot_name_in_model = 'rm_65_description'

    pkg_share = FindPackageShare(package=package_name).find(package_name)
    urdf_model_path = os.path.join(pkg_share, 'config/gazebo_65_gesture.urdf.xacro')
    table_model_path = os.path.join(pkg_share, 'models/gesture_table/model.sdf')
    cube_model_path = os.path.join(pkg_share, 'models/aruco_cube/model.sdf')

    doc = xacro.parse(open(urdf_model_path))
    xacro.process_doc(doc)
    params = {'robot_description': doc.toxml()}

    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen')

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'use_sim_time': True}, params, {'publish_frequency': 15.0}],
        output='screen')

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', robot_name_in_model],
        output='screen')

    spawn_table = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'gesture_table', '-file', table_model_path],
        output='screen')

    spawn_cube_left = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'aruco_cube_left', '-file', cube_model_path, '-x', '0.45', '-y', '0.15', '-z', '0.78'],
        output='screen')

    spawn_cube_right = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'aruco_cube_right', '-file', cube_model_path, '-x', '0.45', '-y', '-0.15', '-z', '0.78'],
        output='screen')

    # spawner 会轮询 controller_manager，比 ros2 control load_controller 更可靠
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    rm_group_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['rm_group_controller', '--controller-manager', '/controller_manager'],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    hand_group_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['hand_group_controller', '--controller-manager', '/controller_manager'],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # 备用：若 spawn 回调时 controller_manager 尚未就绪，8 秒后再试一次
    delayed_controller_spawner = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=[
                    'joint_state_broadcaster',
                    'rm_group_controller',
                    'hand_group_controller',
                    '--controller-manager',
                    '/controller_manager',
                ],
                parameters=[{'use_sim_time': True}],
                output='screen',
            ),
        ],
    )

    close_evt1 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_robot,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )
    close_evt2 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[rm_group_controller_spawner],
        )
    )
    close_evt3 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=rm_group_controller_spawner,
            on_exit=[hand_group_controller_spawner],
        )
    )

    return LaunchDescription([
        close_evt1,
        close_evt2,
        close_evt3,
        delayed_controller_spawner,
        gazebo,
        node_robot_state_publisher,
        spawn_robot,
        spawn_table,
        spawn_cube_left,
        spawn_cube_right,
    ])
