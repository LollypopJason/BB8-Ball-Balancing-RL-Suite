cmake_minimum_required(VERSION 3.10)
project(bb8_jetson_runner CXX)

# Folder holding the exported bb8_policy_inference.cpp / .h
# (GUI: "Export C++ Policy", or cpp/ inside an experiment bundle .zip)
set(POLICY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/policy" CACHE PATH "Folder with bb8_policy_inference.cpp/.h")
option(USE_ONNXRUNTIME "Run the .onnx policy with ONNX Runtime instead of the exported C++" OFF)
option(USE_STEP_DIR "Drive TMC2208s with STEP/DIR pulses (libgpiod) instead of UART VACTUAL" OFF)
set(ONNXRUNTIME_ROOT "/usr/local" CACHE PATH "ONNX Runtime install prefix (include/ + lib/)")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Cortex-A57 (Jetson Nano) tuning when building natively on the board
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=cortex-a57")
endif()
find_package(Threads REQUIRED)

add_executable(bb8_jetson_runner jetson_control_loop.cpp)
target_link_libraries(bb8_jetson_runner Threads::Threads)

if(USE_ONNXRUNTIME)
  target_compile_definitions(bb8_jetson_runner PRIVATE USE_ONNXRUNTIME)
  target_include_directories(bb8_jetson_runner PRIVATE ${ONNXRUNTIME_ROOT}/include
                             ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session)
  target_link_directories(bb8_jetson_runner PRIVATE ${ONNXRUNTIME_ROOT}/lib)
  target_link_libraries(bb8_jetson_runner onnxruntime)
else()
  if(NOT EXISTS "${POLICY_DIR}/bb8_policy_inference.cpp")
    message(FATAL_ERROR "bb8_policy_inference.cpp not found in POLICY_DIR=${POLICY_DIR}. "
                        "Export it from the GUI or pass -DPOLICY_DIR=...")
  endif()
  target_sources(bb8_jetson_runner PRIVATE ${POLICY_DIR}/bb8_policy_inference.cpp)
  target_include_directories(bb8_jetson_runner PRIVATE ${POLICY_DIR})
  # shared library for the Python runner (--cpp-lib)
  add_library(bb8_policy SHARED ${POLICY_DIR}/bb8_policy_inference.cpp)
  target_include_directories(bb8_policy PRIVATE ${POLICY_DIR})
endif()

if(USE_STEP_DIR)
  target_compile_definitions(bb8_jetson_runner PRIVATE USE_STEP_DIR)
  target_link_libraries(bb8_jetson_runner gpiod)
endif()
