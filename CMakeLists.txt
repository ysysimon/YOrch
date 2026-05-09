cmake_minimum_required(VERSION 3.21)

project(YOrch VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(YORCH_BUILD_TESTS "Build YOrch tests" ON)
option(YORCH_BUILD_EXAMPLES "Build YOrch examples" ON)
option(YORCH_BUILD_DOCS "Build YOrch API documentation with Doxygen" OFF)
option(YORCH_BUILD_BENCHMARKS "Build YOrch benchmarks" OFF)
option(YORCH_ENABLE_ASSERTS "Enable YOrch asserts" ON)

add_library(yorch_header_only INTERFACE)
add_library(yorch::yorch_header_only ALIAS yorch_header_only)

target_compile_features(yorch_header_only INTERFACE cxx_std_20)
target_compile_definitions(yorch_header_only INTERFACE
  YORCH_ENABLE_ASSERTS=$<BOOL:${YORCH_ENABLE_ASSERTS}>
)
target_include_directories(yorch_header_only INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_library(yorch
  src/yorch.cpp
)
add_library(yorch::yorch ALIAS yorch)

target_compile_features(yorch PUBLIC cxx_std_20)
target_compile_definitions(yorch PUBLIC
  YORCH_ENABLE_ASSERTS=$<BOOL:${YORCH_ENABLE_ASSERTS}>
)
target_include_directories(yorch PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set(YORCH_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/YOrch")

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/YOrchConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/YOrchConfig.cmake"
  INSTALL_DESTINATION "${YORCH_INSTALL_CMAKEDIR}"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/YOrchConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

install(TARGETS yorch
  EXPORT YOrchTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(TARGETS yorch_header_only
  EXPORT YOrchTargets
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

export(EXPORT YOrchTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/YOrchTargets.cmake"
  NAMESPACE yorch::
)

install(EXPORT YOrchTargets
  FILE YOrchTargets.cmake
  NAMESPACE yorch::
  DESTINATION "${YORCH_INSTALL_CMAKEDIR}"
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/YOrchConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/YOrchConfigVersion.cmake"
  DESTINATION "${YORCH_INSTALL_CMAKEDIR}"
)

if(EXISTS "${PROJECT_SOURCE_DIR}/include")
  install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )
endif()

if(YORCH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(YORCH_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(YORCH_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(YORCH_BUILD_DOCS)
    find_package(Doxygen REQUIRED)

    set(YORCH_DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/target")
    set(YORCH_DOXYFILE_IN "${PROJECT_SOURCE_DIR}/doxygen/Doxyfile.in")
    set(YORCH_DOXYFILE_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

    configure_file("${YORCH_DOXYFILE_IN}" "${YORCH_DOXYFILE_OUT}" @ONLY)

    add_custom_target(docs
        COMMAND "${DOXYGEN_EXECUTABLE}" "${YORCH_DOXYFILE_OUT}"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${PROJECT_SOURCE_DIR}/assets"
            "${YORCH_DOXYGEN_OUTPUT_DIR}/html/assets"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()
