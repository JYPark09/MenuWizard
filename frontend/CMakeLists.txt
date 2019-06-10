# Set cmake requirement
cmake_minimum_required(VERSION 3.8.2 FATAL_ERROR)

# Declare project
project(MenuWizard)

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRETORY ${PROJECT_BINARY_DIR}/bin)

# Includes
include_directories(include)

# Add executable
add_executable(MenuWizard main.cc)