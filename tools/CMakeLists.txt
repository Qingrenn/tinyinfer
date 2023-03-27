find_package(Protobuf REQUIRED)

message(STATUS "${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "${PROTOBUF_INCLUDE_DIR}")

if(PROTOBUF_FOUND)
    protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HEADS onnx.proto)
    add_executable(onnx2tinyinfer onnx2tinyinfer.cpp ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HEADS})
    target_include_directories(onnx2tinyinfer PRIVATE ${PROTOBUF_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(onnx2tinyinfer PRIVATE ${PROTOBUF_LIBRARIES})
else()
    message(WARNING "Protobuf not found, onnx model conveter tool won't be built")
endif()