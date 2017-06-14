project( ImageUnderstanding )
cmake_minimum_required( VERSION 2.8 )
#set ( CMAKE_BUILD_TYPE Release )
#add_definitions ( -Wall )
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package ( Threads )
find_package ( OpenCV REQUIRED )

if(OPENCV_FOUND)
    MESSAGE("OpenCV found.")
    include_directories (
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${OpenCV_LIB_DIR}
	)        
else (OPENCV_FOUND)
    MESSAGE(ERROR "OpenCV missing.")
endif(OPENCV_FOUND)

aux_source_directory(. SRC_LIST)

add_executable(${PROJECT_NAME} ${SRC_LIST}
	#SOURCE
	src/main.cpp
	src/Calltech_Image_Matrix.cpp
	src/KFoldValidation.cpp
  src/ClassifierData.cpp
	
	#HEADER
        include/Calltech_Image_Matrix.h
        include/error_metrics.h
	      include/KFoldValidation.h
        include/ClassifierData.h

)

target_link_libraries ( ${PROJECT_NAME} ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT}
)
