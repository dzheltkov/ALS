cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 11)
project(ALS VERSION 0.9 DESCRIPTION "C++ template implementation for ALS" LANGUAGES CXX)

include(cmake/common.cmake)

include(GNUInstallDirs)

add_subdirectory(external/CXXBLAS)

add_library(als INTERFACE)
target_link_libraries(als INTERFACE BLAS::BLAS_CXX)
target_compile_definitions(als INTERFACE $<$<CONFIG:Debug>:DEBUG>)
target_include_directories(cxxblas INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

add_library(ALS::ALS_CXX ALIAS als)

include(cmake/install.cmake)
