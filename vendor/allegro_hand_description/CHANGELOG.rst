^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package allegro_hand_description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.9.0 (2025-09-15)
------------------
* Fix ament_auto warning about headers install destination
* Contributors: Noel Jimenez

1.8.0 (2025-07-08)
------------------

1.7.0 (2025-06-18)
------------------
* Adapt to changes in play_motion2
* Contributors: davidfernandez

1.6.0 (2025-05-28)
------------------
* Change default value for use_xela parameter to False
* add side parameter
* Contributors: Aina, Aina Irisarri

1.5.1 (2025-04-11)
------------------

1.5.0 (2025-04-10)
------------------
* Adding cartesian launch
* Temporary fix to fix the wobbling of the fingers
* Fix inertia for the allegro hand
* Contributors: thomas.peyrucain, vivianamorlando

1.4.4 (2025-04-03)
------------------

1.4.3 (2025-04-01)
------------------

1.4.2 (2025-03-25)
------------------

1.4.1 (2025-03-17)
------------------
* Fix pid_controllers variable
* Contributors: Aina

1.4.0 (2025-03-03)
------------------

1.3.0 (2025-01-16)
------------------
* Fixing rotatory joint
* Contributors: vivianamorlando

1.2.0 (2024-10-07)
------------------
* Update changelogs
* Contributors: Aina

1.1.0 (2024-09-25)
------------------
* Merge branch 'feat/bhand_controller' into 'main'
  bhand controller
  See merge request device/allegro_hand!19
* fix motion tabs
* add condition and parameter for xela dependency
* add xela sensor
* remove xela dependency from the controller
* create a launch file to use libhand controller
* Add test of controller
* change rviz related files to the description package
* change play_motion2 launch file from simulation pkg to description
* add gazebo
* Contributors: Aina, Aina Irisarri

1.0.0 (2024-08-01)
------------------
* Merge branch 'feat/xela_plugin' into 'main'
  Add xela sensor plugin
  See merge request device/allegro_hand!28
* Add xela sensor plugin
* Merge branch 'fix/package_version' into 'main'
  update package version
  See merge request device/allegro_hand!24
* update package version
* Merge branch 'air/fix/robot_state_publisher' into 'main'
  fix robot description bug
  See merge request device/allegro_hand!20
* fix robot description bug
* Merge branch 'fix/mirror_left_hand' into 'main'
  Interface names now match the ones in the urdf
  See merge request device/allegro_hand!16
* mirror fingers
* all effort & vel same values
* new pids & launch files fixes
* add controller_manager
* reallocate real hand launch file
* real hand launch file & allegro hand pluging added
* tmp
* add pid_controllers arg
* Merge branch 'feat/open_close_test' into 'main'
  open close test
  See merge request device/allegro_hand!13
* remove tip pid
* change rotate axes
* Merge branch 'feat/controllers' into 'main'
  Add controllers in gazebo simulation
  See merge request device/allegro_hand!12
* add tests && their dependencies
* changing gazebo_ros to pal_gazebo_worlds
* update names, add join_state_broadcaster & fix collision
* launch controllers with the gazebo simulation
* Merge branch 'simulation_package' into 'main'
  create simulation package
  See merge request device/allegro_hand!11
* create simulation package & changing config and launch files and dependencies
* Merge branch 'ros2-allegro-simulation' into 'main'
  Ros2 allegro hand description & simulation
  See merge request device/allegro_hand!2
* launch gazebo_ros instead of pal_gazebo_worlds
* add dependencies
* change finger joint names
* Delete build_desc.sh
* restructure urdf
* remove pdf
* add parameter to specify launching rviz & delete the specific rviz launch file
* delete unnecessary model_name argument
* update dependencies
* update license & params
* add pids macro
* update license and comments
* update license & CMake
* delete ros launch file & gazebo world
* change use_sim_time variable
* delete unnecessary comments
* delete ros launch files
* fix tabulations and add all the controllers
* add ros2 control when applying the allegro-hand alone
* change gazebo simulation structure
* add ros2_control & transmission + rename controllers package
* add prefix variable in urdfs
* update urdf structure
* fix diccionary issue with robot description param
* create a rviz launch file without robot_state publisher
* update urdf
* update the origin coordinates
* rviz visualization
* rename .urdf.xacro files
* adding gazebo package
* delete src folder
* Contributors: Aina, Aina Irisarri, Jordan Palacios
