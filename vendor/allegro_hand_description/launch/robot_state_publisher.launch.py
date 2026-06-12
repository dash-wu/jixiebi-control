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

import os

from dataclasses import dataclass

from ament_index_python.packages import get_package_share_directory
from launch_pal.arg_utils import LaunchArgumentsBase
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription
from launch_pal.arg_utils import read_launch_argument
from launch_pal.robot_arguments import CommonArgs
from launch_ros.parameter_descriptions import ParameterValue
from pathlib import Path
from launch_param_builder import load_xacro
from launch_ros.actions import Node


@dataclass(frozen=True)
class LaunchArguments(LaunchArgumentsBase):

    use_sim_time: DeclareLaunchArgument = CommonArgs.use_sim_time

    pid_controllers: DeclareLaunchArgument = DeclareLaunchArgument(
        name='pid_controllers',
        default_value='false',
        choices=['true', 'false'])

    use_xela_sensor: DeclareLaunchArgument = DeclareLaunchArgument(
        name='use_xela_sensor',
        default_value='False',
        choices=['True', 'False'],
        description='Active xela sensors')
    side: DeclareLaunchArgument = DeclareLaunchArgument(
        name='side',
        default_value='right',
        choices=['right', 'left'],
        description=' side')


def launch_setup(context, *args, **kwargs):

    robot_description_content = load_xacro(
        Path(os.path.join(
            get_package_share_directory('allegro_hand_description'), 'robots',
            'palm_hand_test.urdf.xacro')),
        {'use_sim_time': read_launch_argument('use_sim_time', context),
         'pid_controllers': read_launch_argument('pid_controllers', context),
         'use_xela_sensor': read_launch_argument('use_xela_sensor', context),
         'side': read_launch_argument('side', context)}
    )
    robot_description = ParameterValue(robot_description_content, value_type=None)
    # which parses the param as YAML instead of string

    rsp = Node(package='robot_state_publisher',
               executable='robot_state_publisher',
               output='both',
               parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                            'pid_controllers': LaunchConfiguration('pid_controllers'),
                            'use_xela_sensor': LaunchConfiguration('use_xela_sensor'),
                            'side': LaunchConfiguration('side'),
                            'robot_description': robot_description}])
    return [rsp]


def generate_launch_description():

    # Create the launch description and populate
    ld = LaunchDescription()
    launch_arguments = LaunchArguments()

    launch_arguments.add_to_launch_description(ld)

    ld.add_action(OpaqueFunction(function=launch_setup))

    return ld
