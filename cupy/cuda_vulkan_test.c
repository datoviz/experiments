#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

// Buffer size (1MB)
#define BUFFER_SIZE (1024 * 1024)

// External function from CUDA file
extern void launch_cuda_kernel(float* data, int size);

// Check Vulkan result
#define VK_CHECK(x)                                                                               \
    do                                                                                            \
    {                                                                                             \
        VkResult err = x;                                                                         \
        if (err)                                                                                  \
        {                                                                                         \
            fprintf(stderr, "❌ Vulkan Error %d at %s:%d\n", err, __FILE__, __LINE__);            \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

// Check CUDA result
#define CUDA_CHECK(x)                                                                             \
    do                                                                                            \
    {                                                                                             \
        CUresult err = x;                                                                         \
        if (err != CUDA_SUCCESS)                                                                  \
        {                                                                                         \
            fprintf(stderr, "❌ CUDA Error %d at %s:%d\n", err, __FILE__, __LINE__);              \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

int main()
{
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkBuffer buffer;
    VkDeviceMemory memory;
    int fd;

    // --- 1. Vulkan Instance & Device ---
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Vulkan-CUDA Test",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0,
    };

    VkInstanceCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };

    VK_CHECK(vkCreateInstance(&createInfo, NULL, &instance));

    uint32_t deviceCount = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, NULL));
    VkPhysicalDevice devices[deviceCount];
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, devices));
    physicalDevice = devices[0];

    // Enable Vulkan external memory extensions
    const char* deviceExtensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };

    VkDeviceCreateInfo deviceCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueCreateInfo,
        .enabledExtensionCount = 2,
        .ppEnabledExtensionNames = deviceExtensions};

    VK_CHECK(vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device));

    // --- 2. Vulkan Buffer Creation ---
    VkBufferCreateInfo bufferCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = BUFFER_SIZE,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VK_CHECK(vkCreateBuffer(device, &bufferCreateInfo, NULL, &buffer));

    // --- 3. Vulkan Memory Allocation ---
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    uint32_t memoryTypeIndex = (uint32_t)-1;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        {
            memoryTypeIndex = i;
            break;
        }
    }

    if (memoryTypeIndex == (uint32_t)-1)
    {
        fprintf(stderr, "❌ ERROR: No suitable Vulkan memory type found!\n");
        exit(1);
    }

    VkExportMemoryAllocateInfo exportAllocInfo = {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };

    VkMemoryAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = memoryTypeIndex,
        .pNext = &exportAllocInfo,
    };

    VK_CHECK(vkAllocateMemory(device, &allocInfo, NULL, &memory));
    VK_CHECK(vkBindBufferMemory(device, buffer, memory, 0));

    // --- 4. Export Vulkan Memory to CUDA ---
    VkMemoryGetFdInfoKHR getFdInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = memory,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };

    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR =
        (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!vkGetMemoryFdKHR)
    {
        fprintf(
            stderr, "❌ ERROR: Vulkan function vkGetMemoryFdKHR not found. Ensure Vulkan supports "
                    "external memory!\n");
        exit(1);
    }
    VK_CHECK(vkGetMemoryFdKHR(device, &getFdInfo, &fd));

    // --- 5. Import into CUDA ---
    CUdevice cuDevice;
    CUcontext cuContext;
    CUdeviceptr cuPtr;
    CUDA_CHECK(cuInit(0));
    CUDA_CHECK(cuDeviceGet(&cuDevice, 0));
    CUDA_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC cuMemHandleDesc = {
        .type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
        .handle.fd = fd,
        .size = BUFFER_SIZE,
    };

    CUexternalMemory cuExtMem;
    CUDA_CHECK(cuImportExternalMemory(&cuExtMem, &cuMemHandleDesc));

    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc = {0};
    bufferDesc.offset = 0;
    bufferDesc.size = BUFFER_SIZE;
    bufferDesc.flags = 0;

    CUDA_CHECK(cuExternalMemoryGetMappedBuffer(&cuPtr, cuExtMem, &bufferDesc));

    // --- 6. Upload Data Using Vulkan ---
    void* mappedMemory;
    VK_CHECK(vkMapMemory(device, memory, 0, BUFFER_SIZE, 0, &mappedMemory));

    float* data = (float*)mappedMemory;
    for (int i = 0; i < BUFFER_SIZE / sizeof(float); i++)
    {
        data[i] = (float)(i + 1);
    }
    vkUnmapMemory(device, memory);

    // --- 7. Launch CUDA Kernel ---
    int numElements = BUFFER_SIZE / sizeof(float);
    launch_cuda_kernel((float*)cuPtr, numElements);
    CUDA_CHECK(cuCtxSynchronize());

    // --- 8. Download Data Using Vulkan ---
    VK_CHECK(vkMapMemory(device, memory, 0, BUFFER_SIZE, 0, &mappedMemory));
    float* retrievedData = (float*)mappedMemory;

    // --- 9. Validate Computation ---
    for (int i = 0; i < 10; i++)
    {
        printf("Index %d: %f\n", i, retrievedData[i]);
        if (retrievedData[i] != (float)((i + 1) * 2))
        {
            fprintf(
                stderr, "❌ ERROR: Computation failed at index %d (Expected %f, got %f)\n", i,
                (i + 1) * 2.0f, retrievedData[i]);
            exit(1);
        }
    }
    printf("✅ Computation verified!\n");

    vkUnmapMemory(device, memory);

    // Cleanup
    close(fd);
    vkDestroyBuffer(device, buffer, NULL);
    vkFreeMemory(device, memory, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    cuCtxDestroy(cuContext);

    return 0;
}
