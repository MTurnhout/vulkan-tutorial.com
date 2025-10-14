add_executable(VulkanTutorial
	main.cpp
)

target_include_directories(VulkanTutorial PRIVATE
	${Vulkan_INCLUDE_DIR}
)

target_link_libraries(VulkanTutorial PRIVATE
	Vulkan::Vulkan
	glfw
	glm
	stb
)

# Enable warnings
target_compile_options(VulkanTutorial PRIVATE
	$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
	$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)

# Compile shaders
find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin)
if (NOT GLSLC)
	message(FATAL_ERROR, "glslc not found. Install the Vulkan SDK.")
endif()

set(SHADERS_SOURCE_DIR "./Shaders")
set(SHADERS_DEST_DIR "${CMAKE_BINARY_DIR}/bin/Shaders")

file(MAKE_DIRECTORY ${SHADERS_DEST_DIR})

file(GLOB SHADER_FILES CONFIGURE_DEPENDS
	"${SHADERS_SOURCE_DIR}/*.vert"
	"${SHADERS_SOURCE_DIR}/*.frag"
	"${SHADERS_SOURCE_DIR}/*.comp"
	"${SHADERS_SOURCE_DIR}/*.tesc"
	"${SHADERS_SOURCE_DIR}/*.tese"
	"${SHADERS_SOURCE_DIR}/*.geom"
)

set(SHADER_SPV_FILES)

foreach(SHADER_FILE ${SHADER_FILES})
	get_filename_component(FILE_NAME ${SHADER_FILE} NAME)
	set(SPV_OUTPUT "${SHADERS_DEST_DIR}/${FILE_NAME}.spv")
	list(APPEND SHADER_SPV_FILES ${SPV_OUTPUT})

	add_custom_command(
		OUTPUT ${SPV_OUTPUT}
		COMMAND ${GLSLC} ${SHADER_FILE} -o ${SPV_OUTPUT}
		DEPENDS ${SHADER_FILE}
		COMMENT "Compiling shader ${FILE_NAME} to ${SPV_OUTPUT}"
		VERBATIM
	)
endforeach()

add_custom_target(CompileShaders
	DEPENDS ${SHADER_SPV_FILES}
	COMMENT "Compiling all shaders to SPV"
)

add_dependencies(VulkanTutorial CompileShaders)

# Copy Textures
add_custom_target(COPY_TEXTURES
	COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
	${CMAKE_CURRENT_SOURCE_DIR}/Textures
	${CMAKE_BINARY_DIR}/bin/Textures
	COMMENT "Copying textures to output directory"
)

add_dependencies(VulkanTutorial COPY_TEXTURES)
