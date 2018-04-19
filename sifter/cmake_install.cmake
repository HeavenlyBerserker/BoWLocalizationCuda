# Install script for directory: /uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/usr/local")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "0")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES
    "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift/cudaImage.cu"
    "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift/cudaImage.h"
    "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift/cudaSiftH.cu"
    "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift/cudaSiftH.h"
    "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift/matching.cu"
    "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift/cudaSiftD.h"
    "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift/cudaSift.h"
    "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift/cudautils.h"
    "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift/geomFuncs.cpp"
    "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift/mainSift.cpp"
    "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift/cudaSiftD.cu"
    "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift/CMakeLists.txt"
    "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift/Copyright.txt"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/data" TYPE FILE FILES
    "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift/data/left.pgm"
    "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/CudaSift/data/righ.pgm"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
ELSE(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
ENDIF(CMAKE_INSTALL_COMPONENT)

FILE(WRITE "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/${CMAKE_INSTALL_MANIFEST}" "")
FOREACH(file ${CMAKE_INSTALL_MANIFEST_FILES})
  FILE(APPEND "/uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/sifter/${CMAKE_INSTALL_MANIFEST}" "${file}\n")
ENDFOREACH(file)
