import vulkan as vk
import cupy as cp
import numpy as np
import ctypes
import sys

# Detect OS
IS_LINUX = sys.platform == "linux"

# Load CUDA library
cuda = ctypes.CDLL("libcudart.so") if IS_LINUX else ctypes.CDLL("cudart64_110.dll")

# Buffer size
BUFFER_SIZE = 16 * 1024

# Error checking macros
def CUDA_CHECK(result, msg="CUDA error"):
    if result != 0:
        raise RuntimeError(f"❌ {msg}: {result}")

# Vulkan instance creation
app_info = vk.VkApplicationInfo(
    sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pApplicationName="Vulkan-CuPy Test",
    applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
    pEngineName="No Engine",
    engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
    apiVersion=vk.VK_API_VERSION_1_0
)

instance_create_info = vk.VkInstanceCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pApplicationInfo=app_info
)

instance = vk.vkCreateInstance(instance_create_info, None)
if instance is None:
    raise RuntimeError("❌ ERROR: Failed to create Vulkan instance!")

# Enumerate physical devices (GPUs)
physical_devices = vk.vkEnumeratePhysicalDevices(instance)
if not physical_devices:
    raise RuntimeError("❌ ERROR: No Vulkan-compatible GPUs found!")
physical_device = physical_devices[0]

# Enable external memory extensions
############################################################################
device_extensions = [
    vk.VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    vk.VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME if IS_LINUX else vk.VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME
]
############################################################################

queue_create_info = vk.VkDeviceQueueCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    queueFamilyIndex=0,
    queueCount=1,
    pQueuePriorities=[1.0]
)

device_create_info = vk.VkDeviceCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    queueCreateInfoCount=1,
    pQueueCreateInfos=[queue_create_info],
    enabledExtensionCount=len(device_extensions),
    ppEnabledExtensionNames=device_extensions
)

device = vk.vkCreateDevice(physical_device, device_create_info, None)
if device is None:
    raise RuntimeError("❌ ERROR: Failed to create Vulkan device!")

# Create Vulkan buffer
buffer_create_info = vk.VkBufferCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    size=BUFFER_SIZE,
    usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
)

buffer = vk.vkCreateBuffer(device, buffer_create_info, None)
if buffer is None:
    raise RuntimeError("❌ ERROR: Failed to create Vulkan buffer!")

# Allocate Vulkan memory
memory_requirements = vk.vkGetBufferMemoryRequirements(device, buffer)
memory_properties = vk.vkGetPhysicalDeviceMemoryProperties(physical_device)

# Find suitable memory type
memory_type_index = None
for i in range(memory_properties.memoryTypeCount):
    if (memory_requirements.memoryTypeBits & (1 << i)) and \
            (memory_properties.memoryTypes[i].propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) and \
            (memory_properties.memoryTypes[i].propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT):
        memory_type_index = i
        break

if memory_type_index is None:
    raise RuntimeError("❌ ERROR: No suitable Vulkan memory type found!")

############################################################################
export_info = vk.VkExportMemoryAllocateInfo(
    sType=vk.VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
    handleTypes=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
)

alloc_info = vk.VkMemoryAllocateInfo(
    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    allocationSize=memory_requirements.size,
    memoryTypeIndex=memory_type_index,
    pNext=export_info
)
############################################################################

memory = vk.vkAllocateMemory(device, alloc_info, None)
if memory is None:
    raise RuntimeError("❌ ERROR: Failed to allocate Vulkan memory!")

res = vk.vkBindBufferMemory(device, buffer, memory, 0)

# Export Vulkan memory handle to CUDA
############################################################################
get_fd_info = vk.VkMemoryGetFdInfoKHR(
    sType=vk.VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
    memory=memory,
    handleType=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
)

vkGetMemoryFdKHR = vk.vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR")
fd = vkGetMemoryFdKHR(device, get_fd_info)
############################################################################

if fd == -1:
    raise RuntimeError("❌ ERROR: Failed to export Vulkan memory handle!")

# ---- Import Vulkan Memory into CUDA using ctypes ----
class CudaExternalMemoryHandleDescWin32(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_void_p),  # NT handle
        ("name", ctypes.c_void_p)     # Memory object name
    ]

class CudaExternalMemoryHandleUnion(ctypes.Union):
    _fields_ = [
        ("fd", ctypes.c_int),  # File descriptor for OpaqueFd
        ("win32", CudaExternalMemoryHandleDescWin32),
        ("nvSciBufObject", ctypes.c_void_p)  # NvSciBuf handle
    ]

class CudaExternalMemoryHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),  # cudaExternalMemoryHandleType
        ("handle", CudaExternalMemoryHandleUnion),  # Union of handle types
        ("size", ctypes.c_ulonglong),  # Size of the memory allocation
        ("flags", ctypes.c_uint)  # Flags (zero or cudaExternalMemoryDedicated)
    ]

class CudaExternalMemoryBufferDesc(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_ulonglong),  # Offset into the memory object
        ("size", ctypes.c_ulonglong),    # Size of the buffer
        ("flags", ctypes.c_uint)         # Flags (must be zero)
    ]


############################################################################
# Step 1: Import Vulkan memory into CUDA
mem_desc = CudaExternalMemoryHandleDesc(
    type=1,  # cudaExternalMemoryHandleTypeOpaqueFd
    handle=CudaExternalMemoryHandleUnion(fd=fd),
    size=BUFFER_SIZE
)

mem_obj = ctypes.c_void_p()
res = cuda.cudaImportExternalMemory(ctypes.byref(mem_obj), ctypes.byref(mem_desc))
CUDA_CHECK(res, "Failed to import external memory")

# Step 2: Map external memory to CUDA
buffer_desc = CudaExternalMemoryBufferDesc(
    offset=0,
    size=BUFFER_SIZE,
    flags=0
)

cuda_ptr = ctypes.c_void_p()
CUDA_CHECK(cuda.cudaExternalMemoryGetMappedBuffer(ctypes.byref(cuda_ptr), mem_obj, ctypes.byref(buffer_desc)), "Failed to map Vulkan memory to CUDA")

# ---- Create a CuPy array using UnownedMemory ----
cuda_device_ptr = cuda_ptr.value
if cuda_device_ptr is None:
    raise RuntimeError("❌ ERROR: CUDA device pointer is NULL!")

cupy_mem = cp.cuda.UnownedMemory(cuda_device_ptr, BUFFER_SIZE, memory)
cupy_memptr = cp.cuda.MemoryPointer(cupy_mem, 0)
cupy_array = cp.ndarray((BUFFER_SIZE // 4,), dtype=cp.int32, memptr=cupy_memptr)
############################################################################



t = cp.arange(1, 1 + BUFFER_SIZE // 4).astype(cp.int32)
cupy_array[:] = t

# ---- Run CuPy Kernel ----
cupy_array *= 2
cp.cuda.Device(0).synchronize()

# ---- Validate Computation ----
mapped_memory = vk.vkMapMemory(device, memory, 0, BUFFER_SIZE, 0)
if mapped_memory is None:
    raise RuntimeError("❌ ERROR: Vulkan memory mapping failed!")

arr = np.frombuffer(mapped_memory, dtype=np.int32)
np.testing.assert_array_equal(arr, t.get() * 2)

print("✅ Computation verified!")

# Cleanup
vk.vkDestroyBuffer(device, buffer, None)
vk.vkFreeMemory(device, memory, None)
vk.vkDestroyDevice(device, None)
vk.vkDestroyInstance(instance, None)
