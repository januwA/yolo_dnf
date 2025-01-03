cmake_minimum_required(VERSION 3.5)
# set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508") 

project(q VERSION 0.1.0 LANGUAGES CXX)
# set(CMAKE_CXX_MODULE_STD 1) # this is important
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8 /std:c++latest")
  message(STATUS ${CMAKE_CXX_FLAGS})
endif()

set(OpenCV_DIR "D:\\apps\\opencv\\build\\x64\\vc16\\lib")
find_package(OpenCV REQUIRED)

# jsoncpp-1.9.6>cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
# jsoncpp-1.9.6>cmake --build build --config Release
# set(jsoncpp_DIR "D:\\apps\\jsoncpp-1.9.6\\build")
# find_package(jsoncpp REQUIRED)

add_executable(q json.hpp dnf.cpp main.cpp)

# Must be C++23 until CMake merges `import std` as a C++20 extension
# target_compile_features("${PROJECT_NAME}" PRIVATE cxx_std_23) 

# target_include_directories("${PROJECT_NAME}" PUBLIC "D:\\apps\\jsoncpp-1.9.6\\include")
# target_link_libraries("${PROJECT_NAME}" PUBLIC "D:\\apps\\jsoncpp-1.9.6\\build\\lib\\Release\\jsoncpp.lib")
# target_include_directories(${PROJECT_NAME} PUBLIC "D:\\apps\\json-3.11.3\\single_include")

target_link_libraries("${PROJECT_NAME}" ${OpenCV_LIBS})

# 打印所有 OpenCV_ 开始的变量
# get_cmake_property(all_variables VARIABLES)
# list(SORT all_variables)
# foreach(var IN LISTS all_variables)
#   string(REGEX MATCH "^OpenCV_" result "${var}")
#   if(result)
#     message(STATUS "${var}=${${var}}")
#   endif()
# endforeach()

