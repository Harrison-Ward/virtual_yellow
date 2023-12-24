cmake_minimum_required(VERSION 3.10)
project(background_fill VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

# Set the OpenCV directory
set(OpenCV_DIR /usr/local/Cellar/opencv/4.8.1_5/share/OpenCV)

# Find the required packages
find_package(OpenCV REQUIRED)
find_package(Boost 1.65 REQUIRED COMPONENTS filesystem) # Make sure the version matches

# Include directories for Boost, OpenCV, and your include folder
include_directories(${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} include)

# Set the output folder where your program will be created
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

# Add all the source files you want to compile
add_executable(background_fill src/background_fill.cpp src/fill_functions.cpp)

# Link against the Boost and OpenCV libraries
target_link_libraries(background_fill ${Boost_LIBRARIES} ${OpenCV_LIBS})