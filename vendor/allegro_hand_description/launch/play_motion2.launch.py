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
from launch import LaunchDescription
from launch_pal.include_utils import include_scoped_launch_py_description
from launch.actions import DeclareLaunchArgument
from launch_pal.arg_utils import LaunchArgumentsBase
from launch_pal.robot_arguments import CommonArgs


@dataclass(frozen=True)
class LaunchArguments(LaunchArgumentsBase):

    use_sim_time: DeclareLaunchArgument = CommonArgs.use_sim_time


def declare_actions(ld: LaunchDescription, launch_args: LaunchArguments):
    motions_file_path = os.path.join(
        get_package_share_directory('allegro_hand_description'),
        'config', "allegro_motion_realhand.yaml")

    motion_planner_config = os.path.join(
        get_package_share_directory('allegro_hand_description'),
        'config', "allegro_motion_planner.yaml")

    play_motion2 = include_scoped_launch_py_description(
        'play_motion2', ['launch', 'play_motion2.launch.py'],
        launch_arguments={'motions_file': motions_file_path,
                          'motion_planner_config': motion_planner_config,
                          'use_sim_time':  launch_args.use_sim_time})

    ld.add_action(play_motion2)

    return


def generate_launch_description():

    ld = LaunchDescription()
    launch_arguments = LaunchArguments()

    launch_arguments.add_to_launch_description(ld)

    declare_actions(ld, launch_arguments)

    return ld
