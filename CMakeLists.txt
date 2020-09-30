cmake_minimum_required(VERSION 2.7)

find_package(OpenMP)
IF(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ELSE(OPENMP_FOUND)
	Message("No OpenMP found")
ENDIF(OPENMP_FOUND)

INCLUDE(ConfigureANN.cmake)
INCLUDE(ConfigureCLAPACK.cmake)
INCLUDE(ConfigureITK.cmake)

IF(ANN_FOUND AND CLAPACK_FOUND AND ITK_FOUND)
  ADD_SUBDIRECTORY(RMM)
ELSE(ANN_FOUND AND CLAPACK_FOUND AND ITK_FOUND)
  Message("PROBLEM: One of the required libraries not found. RMM will not be compiled.")  
ENDIF(ANN_FOUND AND CLAPACK_FOUND AND ITK_FOUND) 