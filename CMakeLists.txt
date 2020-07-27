cmake_minimum_required(VERSION 3.12)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

project(image)

#include(cmake/build_prefix.cmake)

add_subdirectory(io)
add_subdirectory(face_assessment)