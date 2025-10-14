# Add GLFW as a submodule
set(GLFW_BUILD_EXAMPLES OFF PARENT_SCOPE)
set(GLFW_BUILD_TESTS OFF PARENT_SCOPE)
set(GLFW_BUILD_DOCS OFF PARENT_SCOPE)
set(GLFW_INSTALL OFF PARENT_SCOPE)

add_subdirectory(glfw EXCLUDE_FROM_ALL)

# Add GLM (header-only)
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/glm)

# Add stb (header-only)
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/stb)
