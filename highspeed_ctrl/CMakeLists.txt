cmake_minimum_required(VERSION 3.0.2)
project(highspeed_ctrl)

find_package(Eigen3 REQUIRED)

set(GSL_ROOT_DIR /home/ros2nx/gsl)

find_package(GSL REQUIRED)

if(NOT EIGEN3_FOUND)
  # Fallback to cmake_modules
  find_package(cmake_modules REQUIRED)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})  # Not strictly necessary as Eigen is head only
  # Possibly map additional variables to the EIGEN3_ prefix.
else()
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()

find_package(catkin REQUIRED COMPONENTS  
  nav_msgs  
  roscpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  ackermann_msgs
  std_msgs
  std_srvs
  tf  
  hmcl_msgs  
  vesc_msgs
  dynamic_reconfigure
  roslib
)
find_package(Boost REQUIRED COMPONENTS filesystem)

generate_dynamic_reconfigure_options(
     cfg/test.cfg
)

catkin_package(
  CATKIN_DEPENDS roslib vesc_msgs nav_msgs roscpp std_srvs sensor_msgs geometry_msgs std_msgs hmcl_msgs tf visualization_msgs dynamic_reconfigure
  LIBRARIES
)

include_directories(
  include  
  ${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  include/control_pkg
  include/FORCESNLPsolver/include    
  include/FORCESNLPsolver  
  ${Boost_INCLUDE_DIRS}
  ${GSL_INCLUDE_DIRS}
)

link_directories(${GSL_LIBRARY_DIRS})

add_definitions(-DROSCONSOLE_MIN_SEVERITY=ROSCONSOLE_SEVERITY_NONE)

add_executable(dynamic_reconfigure_node src/pure_pursuit.cpp src/utils.cpp src/trajectory.cpp src/trajectory_manager.cpp src/lowpass_filter.cpp src/main_ctrl.cpp)
add_dependencies(dynamic_reconfigure_node ${PROJECT_NAME}_gencfg)
# target_link_libraries(dynamic_reconfigure_node ${GSL_LIBRARIES} ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES} -ldl -lm ${CMAKE_CURRENT_SOURCE_DIR}/include/FORCESNLPsolver/lib/libFORCESNLPsolver.so)
target_link_libraries(dynamic_reconfigure_node ${GSL_LIBRARIES} ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES} -ldl -lm)
SET(CMAKE_CXX_FLAGS "-O2 -g -Wall ${CMAKE_CXX_FLAGS}")

add_executable(highspeed_ctrl src/utils.cpp src/pure_pursuit.cpp src/trajectory.cpp src/trajectory_manager.cpp src/lowpass_filter.cpp src/main_ctrl.cpp)
add_dependencies(highspeed_ctrl ${catkin_EXPORTED_TARGETS})
# target_link_libraries(highspeed_ctrl ${GSL_LIBRARIES} ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES} -ldl -lm ${CMAKE_CURRENT_SOURCE_DIR}/include/FORCESNLPsolver/lib/libFORCESNLPsolver.so)
target_link_libraries(highspeed_ctrl ${GSL_LIBRARIES} ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES} -ldl -lm )



# install(
#   TARGETS
#   highspeed_ctrl    
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )


# install(DIRECTORY launch/
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
# )
