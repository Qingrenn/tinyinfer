set(TINYINFER_SRCS
    mat.cpp
    allocator.cpp
    mat_pixel.cpp
)

find_package(OpenCV REQUIRED)

if(TINYINFER_SHARED_LIB)
    add_library(tinyinfer SHARED ${TINYINFER_SRCS})
else()
    add_library(tinyinfer STATIC ${TINYINFER_SRCS})
endif()

target_link_directories(tinyinfer PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tinyinfer PUBLIC ${OpenCV_LIBS})
