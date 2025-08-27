#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
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
    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _graphicsPipeline;

    vk::CommandPool _commandPool;
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
        CreateGraphicsPipelines();
        CreateFramebuffers();
        CreateCommandPool();
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
        std::vector<vk::PhysicalDevice> devices = _instance.enumeratePhysicalDevices();
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
            vk::ImageViewCreateInfo createInfo{};
            createInfo.sType = vk::StructureType::eImageViewCreateInfo;
            createInfo.image = _swapChainImages[i];
            createInfo.viewType = vk::ImageViewType::e2D;
            createInfo.format = _swapChainImageFormat;
            createInfo.components.r = vk::ComponentSwizzle::eIdentity;
            createInfo.components.g = vk::ComponentSwizzle::eIdentity;
            createInfo.components.b = vk::ComponentSwizzle::eIdentity;
            createInfo.components.a = vk::ComponentSwizzle::eIdentity;
            createInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            _swapChainImageViews[i] = _device.createImageView(createInfo);
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

    void CreateGraphicsPipelines()
    {
        const auto vertShaderCode = ReadFile("Shaders/shader.vert.spv");
        const auto fragShaderCode = ReadFile("Shaders/shader.frag.spv");

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

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
        // vertexInputInfo.setVertexAttributeDescriptions();
        // vertexInputInfo.setVertexAttributeDescriptions();

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
        rasterizerInfo.frontFace = vk::FrontFace::eClockwise;
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
        // pipelineLayoutInfo.setSetLayouts();
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
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &ClearColor;

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

        commandBuffer.draw(3, 1, 0, 0);
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

        return
            // deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu &&
            // deviceFeatures.geometryShader &&
            indices.IsComplete() && extensionsSupported && swapChainAdequate;
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