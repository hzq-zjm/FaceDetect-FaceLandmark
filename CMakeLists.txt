cmake_minimum_required(VERSION 2.8)

project(FaceLandmark)

add_definitions(-std=c++11)
add_definitions("-Wall")

set(OpenCV_LIB_DIR "/usr/local/lib")
set(OpenCV_INCLUDE_DIR "/usr/local/include")

if(True)
    find_package(OpenMP)
    if(NOT TARGET OpenMP::OpenMP_CXX AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

include_directories(${OpenCV_INCLUDE_DIR}/opencv4
                    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/include/ncnn
                    ./include)

link_directories(${OpenCV_LIB_DIR}
                 ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/lib)
set(OpenCV_LIB_DIR opencv_core 
                opencv_highgui 
                opencv_imgproc 
                opencv_imgcodecs 
                opencv_videoio)

set(SRC
        ./src/main.cpp
        ./src/Pfld.cpp
        ./src/UltraFace.cpp
        )

add_executable(FaceLandmark ${SRC})
target_link_libraries(FaceLandmark ncnn ${OpenCV_LIB_DIR})
if(OpenMP_CXX_FOUND)
    if(NCNN_CMAKE_VERBOSE)
        message("Building with OpenMP")
    endif()
    target_link_libraries(FaceLandmark PUBLIC OpenMP::OpenMP_CXX)
endif()
