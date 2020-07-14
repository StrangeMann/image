set(OpenCV_DIR D:/Papka/myProgramms/opencv/build/x64/vc15/lib)

project( edma_measure )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( edma_measure edma_measure.h edma_measure.cpp )
target_link_libraries( edma_measure ${OpenCV_LIBS} )

set_property(TARGET DisplayImage PROPERTY CXX_STANDARD 17)