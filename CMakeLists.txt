cmake_minimum_required(VERSION 3.5)
project(mgx_inference)

set (CMAKE_CXX_STANDARD 17)
set (SRC inference)

set (CMAKE_BUILD_TYPE Debug)

list (APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package (migraphx REQUIRED)
find_package (hip REQUIRED)

message("source file: " ${SRC}.cpp " ---> bin: " ${SRC})
add_executable(${SRC} ${SRC}.cpp)

target_link_libraries(${SRC}
  migraphx::c
  hip::device
  glog
  pthread)
