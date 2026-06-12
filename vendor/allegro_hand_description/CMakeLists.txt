cmake_minimum_required(VERSION 3.8)
project(allegro_hand_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY meshes robots urdf gazebo ros2_control launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
