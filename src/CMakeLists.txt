set(TINYINFER_SRCS
    mat.cpp
)

if(TINYINFER_SHARED_LIB)
    add_library(tinyinfer SHARED ${TINYINFER_SRCS})
else()
    add_library(tinyinfer STATIC ${TINYINFER_SRCS})
endif()