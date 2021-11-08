project(tensorboard_logger)
cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set (CMAKE_CXX_STANDARD 11)
find_package(Protobuf REQUIRED)

file(GLOB protos "proto/*.proto")

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${protos})

add_library(tensorboard_logger STATIC
    "src/crc.cc"
    "src/tensorboard_logger.cc"
    ${PROTO_SRCS}
)
target_include_directories(tensorboard_logger PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${Protobuf_INCLUDE_DIRS}
    ${PROJECT_BINARY_DIR}
    )
target_link_libraries(tensorboard_logger PUBLIC ${Protobuf_LIBRARIES})

add_executable(test
    tests/test_tensorboard_logger.cc
)
target_link_libraries(test tensorboard_logger)
