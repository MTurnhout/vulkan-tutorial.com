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

static const std::vector<const char *> ValidationLayers =
#ifdef NDEBUG
    {};
#else
    {"VK_LAYER_KHRONOS_validation"};
#endif
static const bool EnableValidationLayers = !ValidationLayers.empty();

const std::vector<const char *> DeviceExtensions = { vk::KHRSwapchainExtensionName };

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

    vk::Instance _instance = nullptr;
    vk::detail::DispatchLoaderDynamic _dispatchLoaderDynamic;
    vk::DebugUtilsMessengerEXT _debugMessenger = nullptr;
    vk::SurfaceKHR _surface = nullptr;

    vk::PhysicalDevice _physicalDevice = nullptr;
    vk::Device _device = nullptr;

    vk::Queue _graphicsQueue = nullptr;
    vk::Queue _presentQueue = nullptr;

    vk::SwapchainKHR _swapChain = nullptr;
    std::vector<vk::Image> _swapChainImages;
    vk::Format _swapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D _swapChainExtent = {};
    std::vector<vk::ImageView> _swapChainImageViews;

    vk::RenderPass _renderPass = nullptr;
    vk::PipelineLayout _pipelineLayout = nullptr;
    vk::Pipeline _graphicsPipeline = nullptr;

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
    }

    void MainLoop() const
    {
        while (!glfwWindowShouldClose(_window))
        {
            glfwPollEvents();
        }
    }

    void Cleanup() const
    {
        _device.destroyPipeline(_graphicsPipeline);
        _device.destroyPipelineLayout(_pipelineLayout);
        _device.destroyRenderPass(_renderPass);

        for (const auto imageView : _swapChainImageViews)
        {
            _device.destroyImageView(imageView);
        }

        _device.destroySwapchainKHR(_swapChain);
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

    void CreateInstance()
    {
        if (EnableValidationLayers && !CheckValidationLayerSupport())
        {
            throw std::runtime_error("Validation layers requested, but not available!");
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
        std::set<uint32_t> uniqueQueueFamilies = {
            graphicsFamily.value(),
            presentFamily.value(),
        };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies)
        {
            vk::DeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = vk::StructureType::eDeviceQueueCreateInfo;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;

            queueCreateInfos.push_back(queueCreateInfo);
        }

        vk::PhysicalDeviceFeatures deviceFeatures{};

        vk::DeviceCreateInfo createInfo{};
        createInfo.sType = vk::StructureType::eDeviceCreateInfo;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.queueCreateInfoCount = 1;

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
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
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
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentReference;

        vk::RenderPassCreateInfo renderPassInfo{};
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

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
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;

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
        viewportStateInfo.viewportCount = 1;
        viewportStateInfo.pViewports = &viewport;
        viewportStateInfo.scissorCount = 1;
        viewportStateInfo.pScissors = &scissor;

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
        colorBlendingInfo.attachmentCount = 1;
        colorBlendingInfo.pAttachments = &colorBlendAttachmentState;
        colorBlendingInfo.blendConstants[0] = 0.0f;
        colorBlendingInfo.blendConstants[1] = 0.0f;
        colorBlendingInfo.blendConstants[2] = 0.0f;
        colorBlendingInfo.blendConstants[3] = 0.0f;

        const std::vector dynamicStates = {
            vk::DynamicState::eViewport, vk::DynamicState::eScissor
        };

        vk::PipelineDynamicStateCreateInfo dynamicStateInfo{};
        dynamicStateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicStateInfo.pDynamicStates = dynamicStates.data();

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.setLayoutCount = 0;
        pipelineLayoutInfo.pSetLayouts = nullptr;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;

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

        std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
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
        vk::DebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
        vk::DebugUtilsMessageTypeFlagsEXT /*messageType*/,
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