#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>

constexpr uint32_t Width = 800;
constexpr uint32_t Height = 600;

static const std::vector<const char *> validationLayers =
#ifdef NDEBUG
    {};
#else
    {"VK_LAYER_KHRONOS_validation"};
#endif
static const bool enableValidationLayers = !validationLayers.empty();

const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

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
        for (const auto imageView : _swapChainImageViews)
        {
            _device.destroyImageView(imageView);
        }

        _device.destroySwapchainKHR(_swapChain);
        _device.destroy();

        if (enableValidationLayers)
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
        if (enableValidationLayers && !CheckValidationLayerSupport())
        {
            throw std::runtime_error("Validation layers requested, but not available!");
        }

        vk::ApplicationInfo appInfo{};
        appInfo.sType = vk::StructureType::eApplicationInfo;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

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
        if (enableValidationLayers)
        {
            createInfo.setPEnabledLayerNames(validationLayers);

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
        if (!enableValidationLayers)
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
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames =
            validationLayers.empty() ? nullptr : validationLayers.data();

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
        for (const auto &requiredExtension : deviceExtensions)
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
        if (enableValidationLayers)
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

        for (const char *layerName : validationLayers)
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