# Vulkan Tutorial
Code following tutorial at:
https://vulkan-tutorial.com

## Project Structure
- `src/VulkanTutorial/`: Main executable project.
- `libs/`: Contains GLFW (tag 3.4) and GLM (tag 1.0.1) submodules.
- `.github/workflows/build-and-release.yml`: GitHub Actions workflow for building and releasing on tagged pushes (`v*`).

## Build Requirements

### General
- **Vulkan SDK**
  - Download: [LunarG Vulkan SDK](https://vulkan.lunarg.com/)
- **CMake**
  - Version: 3.20 or higher
  - Download: [CMake](https://cmake.org/download/)

### Windows
- **Visual Studio 2022 (Build Tools)**
  - Version: 17.x
  - Download: [Visual Studio](https://visualstudio.microsoft.com/downloads/)
  - Workloads: Desktop development with C++

## Setup Instructions

### 1. Clone the repository
Clone the repository and initialize submodules:
```
git clone https://github.com/MTurnhout/vulkan-tutorial.com.git
```
```
cd vulkan-tutorial.com
```
```
git submodule update --init --recursive
```

### 2. Build (Windows)
Generate and build Visual Studio solution:
```cmd
cmake -B ./build -G "Visual Studio 17 2022" -A x64
```
```cmd
cmake --build ./build --config Release
```