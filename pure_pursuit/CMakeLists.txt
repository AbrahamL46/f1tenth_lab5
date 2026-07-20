cmake_minimum_required(VERSION 3.5)
project(pure_pursuit)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/pure_pursuit_node.py
  scripts/waypoint_logger.py
  scripts/scan_noise_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY waypoints
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
