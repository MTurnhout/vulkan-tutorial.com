add_executable(VulkanTutorial
	main.cpp
)

target_include_directories(VulkanTutorial PRIVATE
	${Vulkan_INCLUDE_DIR}
	${GLFW_INCLUDE_DIR}
	${GLM_INCLUDE_DIR}
)

target_link_libraries(VulkanTutorial PRIVATE
	Vulkan::Vulkan
	glfw
)