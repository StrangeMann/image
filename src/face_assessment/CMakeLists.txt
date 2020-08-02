set(OpenCV_DIR D:/Papka/myProgramms/opencv/build/x64/vc15/lib)

project( face_assessment)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( face_assessment face_assessment.h face_assessment.cpp )
target_link_libraries( face_assessment ${OpenCV_LIBS} )

set_property(TARGET face_assessment PROPERTY CXX_STANDARD 17)