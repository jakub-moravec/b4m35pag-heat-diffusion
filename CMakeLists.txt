cmake_minimum_required(VERSION 3.8)
project(diffusion)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        main_skeleton.cpp)

add_executable(diffusion ${SOURCE_FILES})

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)
target_link_libraries(diffusion mpi)
