cmake_minimum_required(VERSION 3.16.3)
project(tensorboard_logger)

option(BUILD_TEST "Build test" OFF)

find_package(Protobuf REQUIRED)

# -----------------------------------------------------------------------------
# Building the tensorboard_logger library
# -----------------------------------------------------------------------------

file(GLOB protos "proto/*.proto")

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${protos})

add_library(tensorboard_logger SHARED
    "src/crc.cc"
    "src/tensorboard_logger.cc"
    ${PROTO_SRCS}
)

target_compile_features(tensorboard_logger PRIVATE cxx_std_11)
target_compile_options(tensorboard_logger PRIVATE -Wall -O2)

target_include_directories(tensorboard_logger
PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROTOBUF_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(tensorboard_logger PUBLIC ${Protobuf_LIBRARIES})

if (BUILD_TEST)
    add_executable(tensorboard_logger_test tests/test_tensorboard_logger.cc)
    target_include_directories(tensorboard_logger_test
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    )
    target_link_libraries(tensorboard_logger_test tensorboard_logger)
endif()

# -----------------------------------------------------------------------------
# Installing the tensorboard_logger library
# -----------------------------------------------------------------------------

install(
    TARGETS tensorboard_logger
    EXPORT tensorboard_logger_Targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    INCLUDES DESTINATION include
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/tensorboard_loggerConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/tensorboard_loggerConfig.cmake"
    INSTALL_DESTINATION
    ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
    EXPORT tensorboard_logger_Targets
    FILE tensorboard_loggerTargets.cmake
    NAMESPACE tensorboard_logger::
    DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
    FILES "${PROJECT_BINARY_DIR}/tensorboard_loggerConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
    DIRECTORY include/
    DESTINATION include
)

install(
    DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/"
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.pb.h"
    PATTERN "CMakeFiles" EXCLUDE
)
