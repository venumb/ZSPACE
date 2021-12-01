# Install script for directory: C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Users/vishu.b/Source/Repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/out/install/x64-Debug")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/unsupported/Eigen/AdolcForward;/unsupported/Eigen/AlignedVector3;/unsupported/Eigen/ArpackSupport;/unsupported/Eigen/AutoDiff;/unsupported/Eigen/BVH;/unsupported/Eigen/EulerAngles;/unsupported/Eigen/FFT;/unsupported/Eigen/IterativeSolvers;/unsupported/Eigen/KroneckerProduct;/unsupported/Eigen/LevenbergMarquardt;/unsupported/Eigen/MatrixFunctions;/unsupported/Eigen/MoreVectorization;/unsupported/Eigen/MPRealSupport;/unsupported/Eigen/NonLinearOptimization;/unsupported/Eigen/NumericalDiff;/unsupported/Eigen/OpenGLSupport;/unsupported/Eigen/Polynomials;/unsupported/Eigen/Skyline;/unsupported/Eigen/SparseExtra;/unsupported/Eigen/SpecialFunctions;/unsupported/Eigen/Splines")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/unsupported/Eigen" TYPE FILE FILES
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/AdolcForward"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/AlignedVector3"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/ArpackSupport"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/AutoDiff"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/BVH"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/EulerAngles"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/FFT"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/IterativeSolvers"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/KroneckerProduct"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/LevenbergMarquardt"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/MatrixFunctions"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/MoreVectorization"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/MPRealSupport"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/NonLinearOptimization"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/NumericalDiff"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/OpenGLSupport"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/Polynomials"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/Skyline"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/SparseExtra"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/SpecialFunctions"
    "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/Splines"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/unsupported/Eigen/src")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/unsupported/Eigen" TYPE DIRECTORY FILES "C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/vishu.b/source/repos/venumb/ZSPACE/cpp/depends/Eigen_Unsupported/out/build/x64-Debug/Eigen/CXX11/cmake_install.cmake")

endif()

