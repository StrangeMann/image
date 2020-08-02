set(OpenCV_DIR D:/Papka/myProgramms/opencv/build/x64/vc15/lib)

project( DisplayImage )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )

set_property(TARGET DisplayImage PROPERTY CXX_STANDARD 17)