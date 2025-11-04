#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define STB_IMAGE_IMPLEMENTATION
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stb_image.h>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>

constexpr uint32_t Width = 800;
constexpr uint32_t Height = 600;

constexpr int MaxFramesInFlight = 2;

const std::vector ValidationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool EnableValidationLayers = false;
#else
constexpr bool EnableValidationLayers = true;
#endif

const std::vector DeviceExtensions = {vk::KHRSwapchainExtensionName};

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    [[nodiscard]] bool IsComplete() const
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails
{
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

struct Vertex
{
    glm::vec2 pos;
    glm::vec3 color;

    static vk::VertexInputBindingDescription GetBindingDescription()
    {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;

        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 2> GetAttributeDescriptions()
    {
        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

struct UniformBufferObject
{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

const std::vector<Vertex> Vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};
const std::vector<uint16_t> Indices = {0, 1, 2, 2, 3, 0};

class HelloTriangleApplication
{
public:
    void Run()
    {
        InitWindow();
        InitVulkan();
        MainLoop();
        Cleanup();
    }

private:
    GLFWwindow *_window = nullptr;

    vk::Instance _instance;
    vk::detail::DispatchLoaderDynamic _dispatchLoaderDynamic;
    vk::DebugUtilsMessengerEXT _debugMessenger;
    vk::SurfaceKHR _surface;

    vk::PhysicalDevice _physicalDevice;
    vk::Device _device;

    vk::Queue _graphicsQueue;
    vk::Queue _presentQueue;

    vk::SwapchainKHR _swapChain;
    std::vector<vk::Image> _swapChainImages;
    vk::Format _swapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D _swapChainExtent;
    std::vector<vk::ImageView> _swapChainImageViews;
    std::vector<vk::Framebuffer> _swapChainFramebuffers;

    vk::RenderPass _renderPass;
    vk::DescriptorSetLayout _descriptorSetLayout;
    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _graphicsPipeline;

    vk::CommandPool _commandPool;

    vk::Image _textureImage;
    vk::DeviceMemory _textureImageMemory;
    vk::ImageView _textureImageView;
    vk::Sampler _textureSampler;

    vk::Buffer _vertexBuffer;
    vk::DeviceMemory _vertexBufferMemory;
    vk::Buffer _indexBuffer;
    vk::DeviceMemory _indexBufferMemory;

    std::vector<vk::Buffer> _uniformBuffers;
    std::vector<vk::DeviceMemory> _uniformBuffersMemory;
    std::vector<void *> _uniformBuffersMapped;

    vk::DescriptorPool _descriptorPool;
    std::vector<vk::DescriptorSet> _descriptorSets;

    std::vector<vk::CommandBuffer> _commandBuffers;

    std::vector<vk::Semaphore> _imageAvailableSemaphores;
    std::vector<vk::Semaphore> _renderFinishedSemaphores;
    std::vector<vk::Fence> _inFlightFences;
    uint32_t _currentFrame = 0;

    bool _framebufferResized = false;

    void InitWindow()
    {
        if (!glfwInit())
        {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        if (!glfwVulkanSupported())
        {
            throw std::runtime_error("Vulkan is not supported by GLFW");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        _window = glfwCreateWindow(Width, Height, "Vulkan", nullptr, nullptr);
        if (!_window)
        {
            throw std::runtime_error("Failed to create GLFW window");
        }

        glfwSetWindowUserPointer(_window, this);
        glfwSetFramebufferSizeCallback(_window, FramebufferResizeCallback);
    }

    static void FramebufferResizeCallback(GLFWwindow *window, int, int)
    {
        const auto app = static_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));
        app->_framebufferResized = true;
    }

    void InitVulkan()
    {
        CreateInstance();
        SetupDebugMessenger();
        CreateSurface();
        PickPhysicalDevice();
        CreateLogicalDevice();
        CreateSwapChain();
        CreateImageViews();
        CreateRenderPass();
        CreateDescriptorSetLayout();
        CreateGraphicsPipeline();
        CreateFramebuffers();
        CreateCommandPool();
        CreateTextureImage();
        CreateTextureImageView();
        CreateTextureSampler();
        CreateVertexBuffer();
        CreateIndexBuffer();
        CreateUniformBuffers();
        CreateDescriptorPool();
        CreateDescriptorSets();
        CreateCommandBuffers();
        CreateSyncObjects();
    }

    void MainLoop()
    {
        while (!glfwWindowShouldClose(_window))
        {
            glfwPollEvents();
            DrawFrame();
        }

        _device.waitIdle();
    }

    void CleanupSwapChain() const
    {
        for (const auto framebuffer : _swapChainFramebuffers)
        {
            _device.destroyFramebuffer(framebuffer);
        }

        for (const auto imageView : _swapChainImageViews)
        {
            _device.destroyImageView(imageView);
        }

        _device.destroySwapchainKHR(_swapChain);
    }

    void Cleanup() const
    {
        CleanupSwapChain();

        _device.destroySampler(_textureSampler);
        _device.destroyImageView(_textureImageView);

        _device.destroyImage(_textureImage);
        _device.freeMemory(_textureImageMemory);

        for (size_t i = 0; i < MaxFramesInFlight; i++)
        {
            _device.destroyBuffer(_uniformBuffers[i]);
            _device.freeMemory(_uniformBuffersMemory[i]);
        }

        _device.destroyDescriptorPool(_descriptorPool);
        _device.destroyDescriptorSetLayout(_descriptorSetLayout);
        _device.destroyBuffer(_indexBuffer);
        _device.freeMemory(_indexBufferMemory);
        _device.destroyBuffer(_vertexBuffer);
        _device.freeMemory(_vertexBufferMemory);
        _device.destroyPipeline(_graphicsPipeline);
        _device.destroyPipelineLayout(_pipelineLayout);
        _device.destroyRenderPass(_renderPass);

        for (size_t i = 0; i < MaxFramesInFlight; i++)
        {
            _device.destroySemaphore(_imageAvailableSemaphores[i]);
            _device.destroySemaphore(_renderFinishedSemaphores[i]);
            _device.destroyFence(_inFlightFences[i]);
        }

        _device.destroyCommandPool(_commandPool);
        _device.destroy();

        if (EnableValidationLayers)
        {
            _instance.destroyDebugUtilsMessengerEXT(
                _debugMessenger, nullptr, _dispatchLoaderDynamic
            );
        }

        _instance.destroySurfaceKHR(_surface);
        _instance.destroy();

        glfwDestroyWindow(_window);

        glfwTerminate();
    }

    void RecreateSwapChain()
    {
        int width = 0, height = 0;
        glfwGetFramebufferSize(_window, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(_window, &width, &height);
            glfwWaitEvents();
        }

        _device.waitIdle();

        CreateSwapChain();
        CreateImageViews();
        CreateFramebuffers();
    }

    void CreateInstance()
    {
        if constexpr (EnableValidationLayers)
        {
            if (!CheckValidationLayerSupport())
            {
                throw std::runtime_error("Validation layers requested, but not available!");
            }
        }

        vk::ApplicationInfo appInfo{};
        appInfo.sType = vk::StructureType::eApplicationInfo;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = vk::makeApiVersion(1, 0, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = vk::makeApiVersion(1, 0, 0, 0);
        appInfo.apiVersion = vk::ApiVersion10;

        const std::vector<const char *> extensions = GetRequiredExtensions();
        if (!CheckRequiredExtensions(extensions))
        {
            throw std::runtime_error("Required Vulkan extensions not found");
        }

        vk::InstanceCreateInfo createInfo{};
        createInfo.sType = vk::StructureType::eInstanceCreateInfo;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.setPEnabledExtensionNames(extensions);

        vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (EnableValidationLayers)
        {
            createInfo.setPEnabledLayerNames(ValidationLayers);

            PopulateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = &debugCreateInfo;
        }
        else
        {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        _instance = vk::createInstance(createInfo);
        _dispatchLoaderDynamic =
            vk::detail::DispatchLoaderDynamic(_instance, vkGetInstanceProcAddr);
    }

    static void PopulateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT &createInfo)
    {
        createInfo.sType = vk::StructureType::eDebugUtilsMessengerCreateInfoEXT;
        createInfo.messageSeverity =
            // vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
        createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                                 vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                                 vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
        createInfo.pfnUserCallback = DebugCallback;
        createInfo.pUserData = nullptr;
    }

    void SetupDebugMessenger()
    {
        if (!EnableValidationLayers)
        {
            return;
        }

        vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
        PopulateDebugMessengerCreateInfo(createInfo);

        _debugMessenger =
            _instance.createDebugUtilsMessengerEXT(createInfo, nullptr, _dispatchLoaderDynamic);
    }

    void CreateSurface()
    {
        VkSurfaceKHR rawSurface;
        if (glfwCreateWindowSurface(_instance, _window, nullptr, &rawSurface) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create window surface!");
        }

        _surface = rawSurface;
    }

    void PickPhysicalDevice()
    {
        const std::vector<vk::PhysicalDevice> devices = _instance.enumeratePhysicalDevices();
        if (devices.empty())
        {
            throw std::runtime_error("Failed to find GPUs with Vulkan support!");
        }

        for (const auto &device : devices)
        {
            if (IsDeviceSuitable(device))
            {
                _physicalDevice = device;
                break;
            }
        }

        if (_physicalDevice == nullptr)
        {
            throw std::runtime_error("Failed to find a suitable GPU!");
        }
    }

    void CreateLogicalDevice()
    {
        auto [graphicsFamily, presentFamily] = FindQueueFamilies(_physicalDevice);

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set uniqueQueueFamilies = {
            graphicsFamily.value(),
            presentFamily.value(),
        };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies)
        {
            vk::DeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = vk::StructureType::eDeviceQueueCreateInfo;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.setQueuePriorities(queuePriority);

            queueCreateInfos.push_back(queueCreateInfo);
        }

        vk::PhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = vk::True;

        vk::DeviceCreateInfo createInfo{};
        createInfo.sType = vk::StructureType::eDeviceCreateInfo;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.setQueueCreateInfos(queueCreateInfos);

        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(DeviceExtensions.size());
        createInfo.ppEnabledExtensionNames = DeviceExtensions.data();

        createInfo.enabledLayerCount = static_cast<uint32_t>(ValidationLayers.size());
        createInfo.ppEnabledLayerNames =
            ValidationLayers.empty() ? nullptr : ValidationLayers.data();

        _device = _physicalDevice.createDevice(createInfo);
        _device.getQueue(graphicsFamily.value(), 0, &_graphicsQueue);
        _device.getQueue(presentFamily.value(), 0, &_presentQueue);
    }

    void CreateSwapChain()
    {
        const auto [surfaceCapabilities, surfaceFormats, presentModes] =
            QuerySwapChainSupport(_physicalDevice);

        const vk::SurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(surfaceFormats);
        const vk::PresentModeKHR presentMode = ChooseSwapPresentMode(presentModes);
        const vk::Extent2D extent = ChooseSwapExtent(surfaceCapabilities);

        uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
        if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount)
        {
            imageCount = surfaceCapabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR createInfo{};
        createInfo.sType = vk::StructureType::eSwapchainCreateInfoKHR;
        createInfo.surface = _surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

        const auto [graphicsFamily, presentFamily] = FindQueueFamilies(_physicalDevice);
        const uint32_t queueFamilyIndices[] = {graphicsFamily.value(), presentFamily.value()};

        if (graphicsFamily != presentFamily)
        {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.setQueueFamilyIndices(queueFamilyIndices);
        }
        else
        {
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        }

        createInfo.preTransform = surfaceCapabilities.currentTransform;
        createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        createInfo.presentMode = presentMode;
        createInfo.clipped = vk::True;
        createInfo.oldSwapchain = nullptr;

        _swapChain = _device.createSwapchainKHR(createInfo);
        _swapChainImages = _device.getSwapchainImagesKHR(_swapChain);
        _swapChainImageFormat = surfaceFormat.format;
        _swapChainExtent = extent;
    }

    void CreateImageViews()
    {
        _swapChainImageViews.resize(_swapChainImages.size());

        for (size_t i = 0; i < _swapChainImages.size(); i++)
        {
            _swapChainImageViews[i] = CreateImageView(_swapChainImages[i], _swapChainImageFormat);
        }
    }

    void CreateRenderPass()
    {
        vk::AttachmentDescription colorAttachment{};
        colorAttachment.format = _swapChainImageFormat;
        colorAttachment.samples = vk::SampleCountFlagBits::e1;
        colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
        colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

        vk::AttachmentReference colorAttachmentReference{};
        colorAttachmentReference.attachment = 0;
        colorAttachmentReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

        vk::SubpassDescription subpass{};
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.setColorAttachments(colorAttachmentReference);

        vk::SubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependency.srcAccessMask = vk::AccessFlagBits::eNone;
        dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

        vk::RenderPassCreateInfo renderPassInfo{};
        renderPassInfo.setAttachments(colorAttachment);
        renderPassInfo.setSubpasses(subpass);
        renderPassInfo.setDependencies(dependency);

        _renderPass = _device.createRenderPass(renderPassInfo);
    }

    void CreateDescriptorSetLayout()
    {
        vk::DescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
        uboLayoutBinding.pImmutableSamplers = nullptr;

        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.setBindings(uboLayoutBinding);

        _descriptorSetLayout = _device.createDescriptorSetLayout(layoutInfo);
    }

    void CreateGraphicsPipeline()
    {
        const std::vector<char> vertShaderCode = ReadFile("Shaders/shader.vert.spv");
        const std::vector<char> fragShaderCode = ReadFile("Shaders/shader.frag.spv");

        const vk::ShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
        const vk::ShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        vk::PipelineShaderStageCreateInfo shaderStages[] = {
            vertShaderStageInfo, fragShaderStageInfo
        };

        vk::VertexInputBindingDescription bindingDescription = Vertex::GetBindingDescription();
        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions =
            Vertex::GetAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.setVertexBindingDescriptions(bindingDescription);
        vertexInputInfo.setVertexAttributeDescriptions(attributeDescriptions);

        vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo{};
        inputAssemblyInfo.topology = vk::PrimitiveTopology::eTriangleList;
        inputAssemblyInfo.primitiveRestartEnable = vk::False;

        vk::Viewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(_swapChainExtent.width);
        viewport.height = static_cast<float>(_swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        vk::Rect2D scissor{};
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent = _swapChainExtent;

        vk::PipelineViewportStateCreateInfo viewportStateInfo{};
        viewportStateInfo.setViewports(viewport);
        viewportStateInfo.setScissors(scissor);

        vk::PipelineRasterizationStateCreateInfo rasterizerInfo{};
        rasterizerInfo.depthClampEnable = vk::False;
        rasterizerInfo.rasterizerDiscardEnable = vk::False;
        rasterizerInfo.polygonMode = vk::PolygonMode::eFill;
        rasterizerInfo.lineWidth = 1.0f;
        rasterizerInfo.cullMode = vk::CullModeFlagBits::eBack;
        rasterizerInfo.frontFace = vk::FrontFace::eCounterClockwise;
        rasterizerInfo.depthBiasEnable = vk::False;
        rasterizerInfo.depthBiasConstantFactor = 0.0f;
        rasterizerInfo.depthBiasClamp = 0.0f;
        rasterizerInfo.depthBiasSlopeFactor = 0.0f;

        vk::PipelineMultisampleStateCreateInfo multisampleInfo{};
        multisampleInfo.sampleShadingEnable = vk::False;
        multisampleInfo.rasterizationSamples = vk::SampleCountFlagBits::e1;
        multisampleInfo.minSampleShading = 1.0f;
        multisampleInfo.pSampleMask = nullptr;
        multisampleInfo.alphaToCoverageEnable = vk::False;
        multisampleInfo.alphaToOneEnable = vk::False;

        vk::PipelineColorBlendAttachmentState colorBlendAttachmentState{};
        colorBlendAttachmentState.colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        colorBlendAttachmentState.blendEnable = vk::False;
        colorBlendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
        colorBlendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eZero;
        colorBlendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
        colorBlendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        colorBlendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eZero;
        colorBlendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;

        vk::PipelineColorBlendStateCreateInfo colorBlendingInfo{};
        colorBlendingInfo.logicOpEnable = vk::False;
        colorBlendingInfo.logicOp = vk::LogicOp::eCopy;
        colorBlendingInfo.setAttachments(colorBlendAttachmentState);
        colorBlendingInfo.blendConstants[0] = 0.0f;
        colorBlendingInfo.blendConstants[1] = 0.0f;
        colorBlendingInfo.blendConstants[2] = 0.0f;
        colorBlendingInfo.blendConstants[3] = 0.0f;

        const std::vector dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

        vk::PipelineDynamicStateCreateInfo dynamicStateInfo{};
        dynamicStateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicStateInfo.pDynamicStates = dynamicStates.data();

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.setSetLayouts(_descriptorSetLayout);
        // pipelineLayoutInfo.setPushConstantRanges();

        _pipelineLayout = _device.createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.setStages(shaderStages);
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
        pipelineInfo.pViewportState = &viewportStateInfo;
        pipelineInfo.pRasterizationState = &rasterizerInfo;
        pipelineInfo.pMultisampleState = &multisampleInfo;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlendingInfo;
        pipelineInfo.pDynamicState = &dynamicStateInfo;
        pipelineInfo.layout = _pipelineLayout;
        pipelineInfo.renderPass = _renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = nullptr;
        pipelineInfo.basePipelineIndex = -1;

        auto [result, pipeline] = _device.createGraphicsPipeline(nullptr, pipelineInfo);
        if (result != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to create graphics pipeline");
        }

        _graphicsPipeline = pipeline;

        _device.destroyShaderModule(vertShaderModule);
        _device.destroyShaderModule(fragShaderModule);
    }

    void CreateFramebuffers()
    {
        _swapChainFramebuffers.resize(_swapChainImageViews.size());

        for (size_t i = 0; i < _swapChainImageViews.size(); i++)
        {
            const vk::ImageView attachments[] = {_swapChainImageViews[i]};

            vk::FramebufferCreateInfo framebufferInfo{};
            framebufferInfo.renderPass = _renderPass;
            framebufferInfo.setAttachments(attachments);
            framebufferInfo.width = _swapChainExtent.width;
            framebufferInfo.height = _swapChainExtent.height;
            framebufferInfo.layers = 1;

            _swapChainFramebuffers[i] = _device.createFramebuffer(framebufferInfo);
        }
    }

    void CreateCommandPool()
    {
        const auto [graphicsFamily, presentFamily] = FindQueueFamilies(_physicalDevice);

        vk::CommandPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        poolInfo.queueFamilyIndex = graphicsFamily.value();

        _commandPool = _device.createCommandPool(poolInfo);
    }

    void CreateTextureImage()
    {
        int texWidth, texHeight, texChannels;
        stbi_uc *pixels =
            stbi_load("Textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        const vk::DeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels)
        {
            throw std::runtime_error("Failed to load texture image");
        }

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        CreateBuffer(
            imageSize, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer, stagingBufferMemory
        );

        void *data = _device.mapMemory(stagingBufferMemory, 0, imageSize);
        memcpy(data, pixels, imageSize);
        _device.unmapMemory(stagingBufferMemory);

        stbi_image_free(pixels);

        CreateImage(
            texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, _textureImage, _textureImageMemory
        );

        TransitionImageLayout(
            _textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal
        );
        CopyBufferToImage(stagingBuffer, _textureImage, texWidth, texHeight);

        TransitionImageLayout(
            _textureImage, vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal
        );

        _device.destroyBuffer(stagingBuffer);
        _device.freeMemory(stagingBufferMemory);
    }

    vk::ImageView CreateImageView(vk::Image image, vk::Format format)
    {
        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.image = image;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        return _device.createImageView(viewInfo);
    }

    void CreateImage(
        const uint32_t width,
        const uint32_t height,
        const vk::Format format,
        const vk::ImageTiling tiling,
        const vk::ImageUsageFlags usage,
        const vk::MemoryPropertyFlags properties,
        vk::Image &image,
        vk::DeviceMemory &imageMemory
    ) const
    {
        vk::ImageCreateInfo imageInfo{};
        imageInfo.imageType = vk::ImageType::e2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = vk::ImageLayout::eUndefined;
        imageInfo.usage = usage;
        imageInfo.sharingMode = vk::SharingMode::eExclusive;
        imageInfo.samples = vk::SampleCountFlagBits::e1;

        image = _device.createImage(imageInfo);

        vk::MemoryRequirements memRequirements;
        _device.getImageMemoryRequirements(image, &memRequirements);

        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);

        imageMemory = _device.allocateMemory(allocInfo);
        _device.bindImageMemory(image, imageMemory, 0);
    }

    void TransitionImageLayout(
        const vk::Image image, const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout
    ) const
    {
        const vk::CommandBuffer commandBuffer = BeginSingleTimeCommands();

        vk::ImageMemoryBarrier barrier{};
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
        barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined &&
            newLayout == vk::ImageLayout::eTransferDstOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlags(0);
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        }
        else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                 newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        }
        else
        {
            throw std::runtime_error("Unsupported layout transition");
        }

        commandBuffer.pipelineBarrier(
            sourceStage, destinationStage, vk::DependencyFlags(0), 0, nullptr, 0, nullptr, 1,
            &barrier
        );

        EndSingleTimeCommands(commandBuffer);
    }

    void CopyBufferToImage(
        const vk::Buffer buffer, const vk::Image image, uint32_t width, uint32_t height
    ) const
    {
        const vk::CommandBuffer commandBuffer = BeginSingleTimeCommands();

        vk::BufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;

        region.setImageOffset({0, 0, 0});
        region.setImageExtent({width, height, 1});

        commandBuffer.copyBufferToImage(
            buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region
        );

        EndSingleTimeCommands(commandBuffer);
    }

    void CreateTextureImageView()
    {
        _textureImageView = CreateImageView(_textureImage, vk::Format::eR8G8B8A8Srgb);
    }

    void CreateTextureSampler()
    {
        vk::PhysicalDeviceProperties properties = _physicalDevice.getProperties();

        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
        samplerInfo.anisotropyEnable = vk::True;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
        samplerInfo.unnormalizedCoordinates = vk::False;
        samplerInfo.compareEnable = vk::False;
        samplerInfo.compareOp = vk::CompareOp::eAlways;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

        _textureSampler = _device.createSampler(samplerInfo);
    }

    void CreateVertexBuffer()
    {
        const vk::DeviceSize bufferSize = sizeof(Vertices[0]) * Vertices.size();

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        CreateBuffer(
            bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer, stagingBufferMemory
        );

        void *data = _device.mapMemory(stagingBufferMemory, 0, bufferSize);
        memcpy(data, Vertices.data(), bufferSize);
        _device.unmapMemory(stagingBufferMemory);

        CreateBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal, _vertexBuffer, _vertexBufferMemory
        );

        CopyBuffer(stagingBuffer, _vertexBuffer, bufferSize);

        _device.destroyBuffer(stagingBuffer);
        _device.freeMemory(stagingBufferMemory);
    }

    void CreateIndexBuffer()
    {
        const vk::DeviceSize bufferSize = sizeof(Indices[0]) * Indices.size();

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        CreateBuffer(
            bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer, stagingBufferMemory
        );

        void *data = _device.mapMemory(stagingBufferMemory, 0, bufferSize);
        memcpy(data, Indices.data(), bufferSize);
        _device.unmapMemory(stagingBufferMemory);

        CreateBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal, _indexBuffer, _indexBufferMemory
        );

        CopyBuffer(stagingBuffer, _indexBuffer, bufferSize);

        _device.destroyBuffer(stagingBuffer);
        _device.freeMemory(stagingBufferMemory);
    }

    void CreateUniformBuffers()
    {
        _uniformBuffers.resize(MaxFramesInFlight);
        _uniformBuffersMemory.resize(MaxFramesInFlight);
        _uniformBuffersMapped.resize(MaxFramesInFlight);
        for (size_t i = 0; i < MaxFramesInFlight; i++)
        {
            constexpr vk::DeviceSize BufferSize = sizeof(UniformBufferObject);
            CreateBuffer(
                BufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent,
                _uniformBuffers[i], _uniformBuffersMemory[i]
            );
            _uniformBuffersMapped[i] = _device.mapMemory(_uniformBuffersMemory[i], 0, BufferSize);
        }
    }

    void CreateBuffer(
        const vk::DeviceSize size,
        const vk::BufferUsageFlags usage,
        const vk::MemoryPropertyFlags properties,
        vk::Buffer &buffer,
        vk::DeviceMemory &bufferMemory
    ) const
    {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;

        buffer = _device.createBuffer(bufferInfo);

        const vk::MemoryRequirements memoryRequirements =
            _device.getBufferMemoryRequirements(buffer);

        vk::MemoryAllocateInfo memoryAllocateInfo{};
        memoryAllocateInfo.allocationSize = memoryRequirements.size;
        memoryAllocateInfo.memoryTypeIndex =
            FindMemoryType(memoryRequirements.memoryTypeBits, properties);

        bufferMemory = _device.allocateMemory(memoryAllocateInfo);

        _device.bindBufferMemory(buffer, bufferMemory, 0);
    }

    [[nodiscard]] vk::CommandBuffer BeginSingleTimeCommands() const
    {
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandPool = _commandPool;
        allocInfo.commandBufferCount = 1;

        const std::vector<vk::CommandBuffer> commandBuffers =
            _device.allocateCommandBuffers(allocInfo);
        const vk::CommandBuffer commandBuffer = commandBuffers[0];

        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

        commandBuffer.begin(beginInfo);

        return commandBuffer;
    }

    void EndSingleTimeCommands(vk::CommandBuffer commandBuffer) const
    {
        commandBuffer.end();

        vk::SubmitInfo submitInfo{};
        submitInfo.setCommandBuffers(commandBuffer);

        _graphicsQueue.submit(submitInfo);
        _graphicsQueue.waitIdle();

        _device.freeCommandBuffers(_commandPool, commandBuffer);
    }

    void CopyBuffer(
        const vk::Buffer srcBuffer, const vk::Buffer dstBuffer, const vk::DeviceSize size
    ) const
    {
        const auto commandBuffer = BeginSingleTimeCommands();

        vk::BufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = size;

        commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

        EndSingleTimeCommands(commandBuffer);
    }

    [[nodiscard]] uint32_t
    FindMemoryType(const uint32_t typeFilter, const vk::MemoryPropertyFlags properties) const
    {
        const vk::PhysicalDeviceMemoryProperties memoryProperties =
            _physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
        {
            if (typeFilter & 1 << i &&
                (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type");
    }

    void CreateDescriptorPool()
    {
        vk::DescriptorPoolSize poolSize{};
        poolSize.type = vk::DescriptorType::eUniformBuffer;
        poolSize.descriptorCount = static_cast<uint32_t>(MaxFramesInFlight);

        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.setPoolSizes(poolSize);
        poolInfo.maxSets = static_cast<uint32_t>(MaxFramesInFlight);

        _descriptorPool = _device.createDescriptorPool(poolInfo);
    }

    void CreateDescriptorSets()
    {
        std::vector<vk::DescriptorSetLayout> layouts(MaxFramesInFlight, _descriptorSetLayout);

        vk::DescriptorSetAllocateInfo allocInfo{};
        allocInfo.descriptorPool = _descriptorPool;
        allocInfo.setSetLayouts(layouts);

        _descriptorSets.resize(MaxFramesInFlight);
        _descriptorSets = _device.allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < MaxFramesInFlight; i++)
        {
            vk::DescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = _uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            vk::WriteDescriptorSet descriptorWrite{};
            descriptorWrite.dstSet = _descriptorSets[i];
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pBufferInfo = &bufferInfo;
            descriptorWrite.pImageInfo = nullptr;
            descriptorWrite.pTexelBufferView = nullptr;

            _device.updateDescriptorSets(descriptorWrite, nullptr);
        }
    }

    void CreateCommandBuffers()
    {
        _commandBuffers.resize(MaxFramesInFlight);

        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = _commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = static_cast<uint32_t>(_commandBuffers.size());

        _commandBuffers = _device.allocateCommandBuffers(allocInfo);
    }

    void RecordCommandBuffer(const vk::CommandBuffer commandBuffer, const uint32_t imageIndex) const
    {
        vk::CommandBufferBeginInfo beginInfo{};
        // beginInfo.flags = vk::CommandBufferUsageFlags::BitsType::eOneTimeSubmit;
        beginInfo.pInheritanceInfo = nullptr;

        commandBuffer.begin(beginInfo);

        constexpr vk::ClearValue ClearColor{{0.0f, 0.0f, 0.0f, 1.0f}};

        vk::RenderPassBeginInfo renderPassInfo{};
        renderPassInfo.renderPass = _renderPass;
        renderPassInfo.framebuffer = _swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset.x = 0;
        renderPassInfo.renderArea.offset.y = 0;
        renderPassInfo.renderArea.extent = _swapChainExtent;
        renderPassInfo.setClearValues(ClearColor);

        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _graphicsPipeline);

        vk::Viewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(_swapChainExtent.width);
        viewport.height = static_cast<float>(_swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        commandBuffer.setViewport(0, 1, &viewport);

        vk::Rect2D scissor{};
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent = _swapChainExtent;

        commandBuffer.setScissor(0, 1, &scissor);

        const vk::Buffer vertexBuffers[] = {_vertexBuffer};
        constexpr vk::DeviceSize Offsets[] = {0};
        commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, Offsets);

        commandBuffer.bindIndexBuffer(_indexBuffer, 0, vk::IndexType::eUint16);

        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, _pipelineLayout, 0, 1,
            &_descriptorSets[_currentFrame], 0, nullptr
        );

        commandBuffer.drawIndexed(static_cast<uint32_t>(Indices.size()), 1, 0, 0, 0);
        commandBuffer.endRenderPass();
        commandBuffer.end();
    }

    void CreateSyncObjects()
    {
        _imageAvailableSemaphores.resize(MaxFramesInFlight);
        _renderFinishedSemaphores.resize(MaxFramesInFlight);
        _inFlightFences.resize(MaxFramesInFlight);

        constexpr vk::SemaphoreCreateInfo SemaphoreInfo{};
        vk::FenceCreateInfo fenceInfo{};
        fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

        for (size_t i = 0; i < MaxFramesInFlight; i++)
        {
            _imageAvailableSemaphores[i] = _device.createSemaphore(SemaphoreInfo);
            _renderFinishedSemaphores[i] = _device.createSemaphore(SemaphoreInfo);
            _inFlightFences[i] = _device.createFence(fenceInfo);
        }
    }

    void UpdateUniformBuffer(const uint32_t currentImage) const
    {
        static auto startTime = std::chrono::high_resolution_clock::now();

        const auto currentTime = std::chrono::high_resolution_clock::now();
        const float time =
            std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime)
                .count();

        UniformBufferObject ubo{};
        ubo.model =
            glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(
            glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)
        );
        ubo.proj = glm::perspective(
            glm::radians(45.0f),
            static_cast<float>(_swapChainExtent.width) /
                static_cast<float>(_swapChainExtent.height),
            0.1f, 10.0f
        );
        ubo.proj[1][1] *= -1;

        memcpy(_uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void DrawFrame()
    {
        if (_device.waitForFences(1, &_inFlightFences[_currentFrame], vk::True, UINT64_MAX) !=
            vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to wait for draw frame fence");
        }

        auto [acquireNextImageKhrResult, imageIndex] = _device.acquireNextImageKHR(
            _swapChain, UINT64_MAX, _imageAvailableSemaphores[_currentFrame], nullptr
        );
        if (acquireNextImageKhrResult == vk::Result::eErrorOutOfDateKHR)
        {
            RecreateSwapChain();
            return;
        }
        if (acquireNextImageKhrResult != vk::Result::eSuccess &&
            acquireNextImageKhrResult != vk::Result::eSuboptimalKHR)
        {
            throw std::runtime_error("Failed to acquire image from swap chain");
        }

        UpdateUniformBuffer(_currentFrame);

        if (_device.resetFences(1, &_inFlightFences[_currentFrame]) != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to reset draw frame fence");
        }

        _commandBuffers[_currentFrame].reset();

        RecordCommandBuffer(_commandBuffers[_currentFrame], imageIndex);

        const vk::Semaphore waitSemaphore[] = {_imageAvailableSemaphores[_currentFrame]};
        constexpr vk::PipelineStageFlags WaitStages[] = {
            vk::PipelineStageFlagBits::eColorAttachmentOutput
        };
        const vk::Semaphore signalSemaphore[] = {_renderFinishedSemaphores[_currentFrame]};

        vk::SubmitInfo submitInfo{};
        submitInfo.setWaitSemaphores(waitSemaphore);
        submitInfo.setWaitDstStageMask(WaitStages);
        submitInfo.setCommandBuffers(_commandBuffers[_currentFrame]);
        submitInfo.setSignalSemaphores(signalSemaphore);

        if (_graphicsQueue.submit(1, &submitInfo, _inFlightFences[_currentFrame]) !=
            vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to submit draw command buffer");
        }

        const vk::SwapchainKHR swapChains[] = {_swapChain};
        vk::PresentInfoKHR presentInfo{};
        presentInfo.setWaitSemaphores(signalSemaphore);
        presentInfo.setSwapchains(swapChains);
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        if (const vk::Result presentKhrResult = _presentQueue.presentKHR(presentInfo);
            presentKhrResult == vk::Result::eErrorOutOfDateKHR ||
            presentKhrResult == vk::Result::eSuboptimalKHR || _framebufferResized)
        {
            _framebufferResized = false;
            RecreateSwapChain();
        }
        else if (presentKhrResult != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to submit request to present image to swap chain");
        }

        _currentFrame = (_currentFrame + 1) % MaxFramesInFlight;
    }

    [[nodiscard]] vk::ShaderModule CreateShaderModule(const std::vector<char> &code) const
    {
        vk::ShaderModuleCreateInfo createInfo{};
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

        return _device.createShaderModule(createInfo);
    }

    static vk::SurfaceFormatKHR
    ChooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats)
    {
        for (const auto &availableFormat : availableFormats)
        {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
                availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    static vk::PresentModeKHR
    ChooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes)
    {
        for (const auto &availablePresentMode : availablePresentModes)
        {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox)
            {
                return availablePresentMode;
            }
        }

        return vk::PresentModeKHR::eFifo;
    }

    [[nodiscard]] vk::Extent2D
    ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) const
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }

        int width, height;
        glfwGetWindowSize(_window, &width, &height);

        vk::Extent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
        };

        actualExtent.width = std::clamp(
            actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width
        );
        actualExtent.height = std::clamp(
            actualExtent.height, capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height
        );

        return actualExtent;
    }

    [[nodiscard]] SwapChainSupportDetails
    QuerySwapChainSupport(const vk::PhysicalDevice device) const
    {
        SwapChainSupportDetails details{};
        details.capabilities = device.getSurfaceCapabilitiesKHR(_surface);
        details.formats = device.getSurfaceFormatsKHR(_surface);
        details.presentModes = device.getSurfacePresentModesKHR(_surface);

        return details;
    }

    [[nodiscard]] bool IsDeviceSuitable(const vk::PhysicalDevice device) const
    {
        // vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
        // vk::PhysicalDeviceFeatures deviceFeatures = device.getFeatures();

        const QueueFamilyIndices indices = FindQueueFamilies(device);

        const bool extensionsSupported = CheckDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported)
        {
            const SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(device);
            swapChainAdequate =
                !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        vk::PhysicalDeviceFeatures supportedFeatures = device.getFeatures();

        return
            // deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu &&
            // deviceFeatures.geometryShader &&
            indices.IsComplete() && extensionsSupported && swapChainAdequate &&
            supportedFeatures.samplerAnisotropy;
    }

    static bool CheckDeviceExtensionSupport(const vk::PhysicalDevice device)
    {
        const std::vector<vk::ExtensionProperties> availableExtensions =
            device.enumerateDeviceExtensionProperties();

        bool requiredExtensionsFound = true;
        for (const auto &requiredExtension : DeviceExtensions)
        {
            bool found = false;
            for (const auto &extension : availableExtensions)
            {
                if (std::strcmp(extension.extensionName, requiredExtension) == 0)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                requiredExtensionsFound = false;
                std::cout << "Required device extension not found: " << requiredExtension
                          << std::endl;
            }
        }

        return requiredExtensionsFound;
    }

    [[nodiscard]] QueueFamilyIndices FindQueueFamilies(const vk::PhysicalDevice device) const
    {
        QueueFamilyIndices indices{};

        const std::vector<vk::QueueFamilyProperties> queueFamilies =
            device.getQueueFamilyProperties();

        int i = 0;
        for (const auto &queueFamily : queueFamilies)
        {
            if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
            {
                indices.graphicsFamily = i;
            }

            if (device.getSurfaceSupportKHR(i, _surface))
            {
                indices.presentFamily = i;
            }

            if (indices.IsComplete())
            {
                break;
            }

            i++;
        }

        return indices;
    }

    static std::vector<const char *> GetRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (EnableValidationLayers)
        {
            extensions.push_back(vk::EXTDebugUtilsExtensionName);
        }

        return extensions;
    }

    static bool CheckRequiredExtensions(const std::vector<const char *> &requiredExtensions)
    {
        const std::vector<vk::ExtensionProperties> extensions =
            vk::enumerateInstanceExtensionProperties();

        bool requiredExtensionsFound = true;
        for (const auto &requiredExtension : requiredExtensions)
        {
            bool found = false;
            for (const auto &extension : extensions)
            {
                if (std::strcmp(extension.extensionName, requiredExtension) == 0)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                requiredExtensionsFound = false;
                std::cout << "Required extension not found: " << requiredExtension << std::endl;
            }
        }

        return requiredExtensionsFound;
    }

    static bool CheckValidationLayerSupport()
    {
        const std::vector<vk::LayerProperties> availableLayers =
            vk::enumerateInstanceLayerProperties();

        for (const char *layerName : ValidationLayers)
        {
            bool layerFound = false;

            for (const auto &layerProperties : availableLayers)
            {
                if (std::strcmp(layerName, layerProperties.layerName) == 0)
                {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound)
            {
                return false;
            }
        }

        return true;
    }

    static std::vector<char> ReadFile(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        const std::streampos fileEndPosition = file.tellg();
        if (fileEndPosition < 0)
        {
            throw std::runtime_error("Failed to determine current read position: " + filename);
        }

        const auto fileSize = static_cast<std::streamsize>(fileEndPosition);
        std::vector<char> buffer(static_cast<size_t>(fileSize));

        file.seekg(0, std::ios::beg);
        if (fileSize > 0 && !file.read(buffer.data(), fileSize))
        {
            throw std::runtime_error("Failed to read file: " + filename);
        }

        return buffer;
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL DebugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT,
        vk::DebugUtilsMessageTypeFlagsEXT,
        const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData,
        void * /*pUserData*/
    )
    {
        if (pCallbackData == nullptr)
        {
            throw std::runtime_error("failed to get callback data");
        }

        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return vk::False;
    }
};

int main()
{
    HelloTriangleApplication app{};

    try
    {
        app.Run();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}