cmake_minimum_required(VERSION 3.21)
set(CMAKE_OSX_SYSROOT /Library/Developer/CommandLineTools/SDKs/MacOSX12.1.sdk/)

project(INFOMCV_A2)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(GLUT REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenGL REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${BOOST_INCLUDE_DIR})


include_directories(src)
include_directories(src/controllers)
include_directories(src/utilities)
add_executable(INFOMCV_A2
        src/controllers/arcball.cpp
        src/controllers/arcball.h
        src/controllers/Camera.cpp
        src/controllers/Camera.h
        src/controllers/Glut.cpp
        src/controllers/Glut.h
        src/controllers/Reconstructor.cpp
        src/controllers/Reconstructor.h
        src/controllers/Scene3DRenderer.cpp
        src/controllers/Scene3DRenderer.h
        src/utilities/General.cpp
        src/utilities/General.h
        src/main.cpp
        src/VoxelReconstruction.cpp
        src/VoxelReconstruction.h)

target_link_libraries (${CMAKE_PROJECT_NAME} ${OPENGL_LIBRARIES} )
# target_link_libraries (${CMAKE_PROJECT_NAME} /Users/mervegurbuz/Library/Frameworks/GLUT.framework)
target_link_libraries (${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries (${CMAKE_PROJECT_NAME} ${OpenMP_LIBRARIES})
target_link_libraries (${CMAKE_PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} stdc++ "-framework OpenGL" "-framework GLUT")
target_link_libraries (${CMAKE_PROJECT_NAME})