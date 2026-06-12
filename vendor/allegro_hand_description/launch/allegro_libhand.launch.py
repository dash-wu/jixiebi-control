# Copyright (c) 2024 PAL Robotics S.L. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from launch.substitutions import PathJoinSubstitution
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_pal.include_utils import include_scoped_launch_py_description
from launch_pal.arg_utils import LaunchArgumentsBase
from launch_pal.include_utils import include_launch_py_description

from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


@dataclass(frozen=True)
class LaunchArguments(LaunchArgumentsBase):

    use_sim_time: DeclareLaunchArgument = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='False',
        description='Use simulation time')

    pid_controllers: DeclareLaunchArgument = DeclareLaunchArgument(
        name='pid_controllers',
        default_value='true',
        choices=['true', 'false'])

    use_xela_sensor: DeclareLaunchArgument = DeclareLaunchArgument(
        name='use_xela_sensor',
        default_value='False',
        choices=['True', 'False'],
        description='Active xela sensors')


def declare_actions(ld: LaunchDescription, launch_args: LaunchArguments):

    robot_state_publisher = include_scoped_launch_py_description(
        pkg_name='allegro_hand_description',
        paths=['launch', 'robot_state_publisher.launch.py'],
        launch_arguments={
            "use_sim_time": launch_args.use_sim_time,
            "pid_controllers": launch_args.pid_controllers,
            "use_xela_sensor": launch_args.use_xela_sensor
        })

    joint_state_broadcaster_launch = include_launch_py_description(
        pkg_name='allegro_hand_controller_configuration',
        paths=['launch', 'joint_state_broadcaster.launch.py'])

    allegro_hand_controllers = include_scoped_launch_py_description(
        pkg_name='allegro_hand_controller_configuration',
        paths=['launch', 'allegro_hand_controller_libhand.launch.py']
    )

    controller_manager_config = PathJoinSubstitution(
        [FindPackageShare(['allegro_hand_controller_configuration']),
         'config', 'controller_manager_cfg.yaml'])

    # ros2 control node
    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[controller_manager_config],
        emulate_tty=True,
        output={
            'stdout': 'screen',
            'stderr': 'screen',
        },
        remappings=[('~/robot_description', '/robot_description')]
    )

    ld.add_action(robot_state_publisher)
    ld.add_action(allegro_hand_controllers)
    ld.add_action(control_node)
    ld.add_action(joint_state_broadcaster_launch)


def generate_launch_description():

    ld = LaunchDescription()

    launch_arguments = LaunchArguments()

    launch_arguments.add_to_launch_description(ld)

    declare_actions(ld, launch_arguments)

    return ld
