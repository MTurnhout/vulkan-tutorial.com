cmake_minimum_required(VERSION 3.27)
project(VulkanTutorial LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/libs)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/libs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)

# Find Vulkan
find_package(Vulkan REQUIRED)
if(NOT Vulkan_FOUND)
	message(FATAL_ERROR "Vulkan not found. Install the Vulkan SDK.")
endif()

# Add libraries
add_subdirectory(libs)

# Add projects
add_subdirectory(src/VulkanTutorial)

# Set startup project (Visual Studio)
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT VulkanTutorial)

# Find ClangFormat
find_program(CLANG_FORMAT_EXECUTABLE NAMES clang-format)
if (CLANG_FORMAT_EXECUTABLE)
	message(STATUS "ClangFormat found: ${CLANG_FORMAT_EXECUTABLE}")

	# Gather all source files across the project
	file(GLOB_RECURSE SOURCES
		"${CMAKE_SOURCE_DIR}/src/*.cpp"
		"${CMAKE_SOURCE_DIR}/src/*.h"
		"${CMAKE_SOURCE_DIR}/src/*.hpp"
	)

	# Add custom target for formatting
	add_custom_target(format
		COMMAND ${CLANG_FORMAT_EXECUTABLE} -i --style=file ${SOURCES}
		COMMENT "Running ClangFormat on all source files"
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	)
else()
	message(WARNING "ClangFormat not found. Formatting target will not be available.")
endif()
