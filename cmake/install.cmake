
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
        TARGETS als
        EXPORT ALSTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

set(ALS_CMAKE_PACKAGE_INSTALL_SUBDIR "share/ALS/cmake")

install(
        EXPORT ALSTargets
        NAMESPACE ::
        DESTINATION ${ALS_CMAKE_PACKAGE_INSTALL_SUBDIR})

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
                                 ${CMAKE_CURRENT_BINARY_DIR}/ALSConfigVersion.cmake
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY SameMinorVersion)

configure_package_config_file(
                              "${PROJECT_SOURCE_DIR}/cmake/ALSConfig.cmake.in" ${CMAKE_CURRENT_BINARY_DIR}/ALSConfig.cmake
                              INSTALL_DESTINATION ${ALS_CMAKE_PACKAGE_INSTALL_SUBDIR})

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ALSConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/ALSConfigVersion.cmake"
        DESTINATION ${ALS_CMAKE_PACKAGE_INSTALL_SUBDIR})
