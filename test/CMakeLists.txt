macro(tinyinfer_add_test name)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE tinyinfer)
    add_test(NAME test_${name} COMMAND ${CMAKE_COMMAND} -DTEST_EXECUTABLE=$<TARGET_FILE:test_${name}> -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/run_test.cmake)
    set_property(TARGET test_${name} PROPERTY FOLDER "tests")
endmacro(tinyinfer_add_test name)

tinyinfer_add_test(mat)
tinyinfer_add_test(mat_pixel)
