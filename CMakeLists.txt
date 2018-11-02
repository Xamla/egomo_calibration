cmake_minimum_required(VERSION 2.8.11)

project(egomo_calibration_py)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(Boost 1.47.0 REQUIRED COMPONENTS system thread chrono)


# Some packages needed for Ceres:
# ===============================
find_package(Eigen REQUIRED)
if (EIGEN_FOUND)
  if (EIGEN_VERSION VERSION_LESS 3.1.0)
    message(FATAL_ERROR "-- Ceres requires Eigen version >= 3.1.0 in order "
      "that Eigen/SparseCore be available, detected version of Eigen is: "
      "${EIGEN_VERSION}")
  endif (EIGEN_VERSION VERSION_LESS 3.1.0)
  message("-- Found Eigen version ${EIGEN_VERSION}: ${EIGEN_INCLUDE_DIRS}")
endif (EIGEN_FOUND)

find_package(LAPACK QUIET)
if (LAPACK_FOUND)
  message("-- Found LAPACK library: ${LAPACK_LIBRARIES}")
else (LAPACK_FOUND)
  message("-- Did not find LAPACK library, disabling LAPACK support.")
endif (LAPACK_FOUND)

find_package(BLAS QUIET)
if (BLAS_FOUND)
  message("-- Found BLAS library: ${BLAS_LIBRARIES}")
else (BLAS_FOUND)
  message("-- Did not find BLAS library, disabling LAPACK support.")
endif (BLAS_FOUND)

find_package(SuiteSparse QUIET)
if (SuiteSparse_FOUND)
  message("-- Found SuiteSparse libraries: ${SUITESPARSE_LIBRARIES} ${SUITESPARSE_VERSION} ${SUITESPARSE_INCLUDE_DIRS}")
  message("-- Found SuiteSparse version ${SUITESPARSE_VERSION} in: ${SUITESPARSE_INCLUDE_DIRS}")
else (SuiteSparse_FOUND)
  message("-- Did not find SuiteSparse libraries, disabling SuiteSparse support.")
endif (SuiteSparse_FOUND)

# Multithreading using OpenMP
option(OPENMP "Enable threaded solving in Ceres (requires OpenMP)" ON)

if (OPENMP)
  # Clang does not (yet) support OpenMP.
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    update_cache_variable(OPENMP OFF)
    message("-- Compiler is Clang, disabling OpenMP.")
    list(APPEND CERES_COMPILE_OPTIONS CERES_NO_THREADS)
  else (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Find quietly s/t as we can continue without OpenMP if it is not found.
    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
      message("-- Building with OpenMP.")
      list(APPEND CERES_COMPILE_OPTIONS CERES_USE_OPENMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      if (UNIX)
        # At least on Linux, we need pthreads to be enabled for mutex to
        # compile.  This may not work on Windows or Android.
        find_package(Threads REQUIRED)
        list(APPEND CERES_COMPILE_OPTIONS CERES_HAVE_PTHREAD)
        list(APPEND CERES_COMPILE_OPTIONS CERES_HAVE_RWLOCK)
      endif (UNIX)
    else (OPENMP_FOUND)
      message("-- Failed to find OpenMP, disabling.")
      update_cache_variable(OPENMP OFF)
      list(APPEND CERES_COMPILE_OPTIONS CERES_NO_THREADS)
    endif (OPENMP_FOUND)
  endif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
else (OPENMP)
  message("-- Building without OpenMP (disabling multithreading).")
  list(APPEND CERES_COMPILE_OPTIONS CERES_NO_THREADS)
endif (OPENMP)

find_package(Glog)
if (GLOG_FOUND)
  message("-- Found Google Log header in: ${GLOG_INCLUDE_DIRS}")
else (GLOG_FOUND)
  message(FATAL_ERROR "Can't find Google Log. Please set GLOG_INCLUDE_DIR & "
    "GLOG_LIBRARY or enable MINIGLOG option to use minimal glog "
    "implementation.")
endif (GLOG_FOUND)

find_package(Ceres REQUIRED)
# ===============================

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fmax-errors=3 -Wall" )

set(src "${SOURCE_DIR}/bundle_adjuster.cpp")

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${SUITESPARSE_INCLUDE_DIRS}
)
include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})
include_directories(SYSTEM ${SUITESPARSE_INCLUDE_DIRS})

link_directories(
  ${Boost_LIBRARY_DIRS}
  #${SUITESPARSE_LIBRARY_DIRS}
)

add_library(egomo_calibration_py MODULE ${src})

target_link_libraries(egomo_calibration_py ceres ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${SUITESPARSE_LIBRARIES} ${GLOG_LIBRARIES} ${Boost_LIBRARIES})

set(INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/usr/local/lib/)
install(TARGETS egomo_calibration_py LIBRARY DESTINATION ${INSTALL_PATH})
#install(DIRECTORY "python/" DESTINATION "${INSTALL_PATH}/egomo_calibration_py" FILES_MATCHING PATTERN "*.py")
