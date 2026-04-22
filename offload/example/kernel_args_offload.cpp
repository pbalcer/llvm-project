#include <offload/OffloadAPI.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define OL_CHECK(expr)                                                         \
  do {                                                                         \
    ol_result_t _r = (expr);                                                   \
    if (_r != OL_SUCCESS) {                                                    \
      std::fprintf(stderr, "%s failed: code=%d details=%s\n", #expr,           \
                   _r->Code, _r->Details ? _r->Details : "");                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static bool pickDevicesCb(ol_device_handle_t Dev, void *UserData) {
  ol_device_handle_t *Output = static_cast<ol_device_handle_t *>(UserData);

  ol_device_type_t Type;
  if (olGetDeviceInfo(Dev, OL_DEVICE_INFO_TYPE, sizeof(Type), &Type) !=
      OL_SUCCESS)
    return true;

  if (Type == OL_DEVICE_TYPE_GPU && *Output == nullptr)
    *Output = Dev;

  return true;
}

static std::vector<char> readFile(const char *Path) {
  std::ifstream F(Path, std::ios::binary | std::ios::ate);
  if (!F) {
    std::fprintf(stderr, "Cannot open %s\n", Path);
    std::exit(1);
  }
  std::streamsize Size = F.tellg();
  F.seekg(0, std::ios::beg);
  std::vector<char> Buf(Size);
  if (!F.read(Buf.data(), Size)) {
    std::fprintf(stderr, "Failed to read %s\n", Path);
    std::exit(1);
  }
  return Buf;
}

int main(int Argc, char **Argv) {
  if (Argc < 3) {
    std::fprintf(stderr, "Usage: %s <kernel-binary> <kernel-name>\n", Argv[0]);
    return 1;
  }
  const char *KernelFile = Argv[1];
  const char *KernelName = Argv[2];

  auto Binary = readFile(KernelFile);

  OL_CHECK(olInit(nullptr));

  ol_device_handle_t Gpu = nullptr;
  OL_CHECK(olIterateDevices(pickDevicesCb, &Gpu));
  if (!Gpu) {
    std::fprintf(stderr, "No GPU device found.\n");
    return 1;
  }

  char DeviceName[256] = {};
  olGetDeviceInfo(Gpu, OL_DEVICE_INFO_NAME, sizeof(DeviceName),
                  DeviceName);
  std::printf("Device: %s\n", DeviceName);

  ol_queue_handle_t Queue = nullptr;
  OL_CHECK(olCreateQueue(Gpu, &Queue));

  ol_program_handle_t Program = nullptr;
  OL_CHECK(olCreateProgram(Gpu, Binary.data(), Binary.size(), &Program));

  ol_symbol_handle_t Kernel = nullptr;
  OL_CHECK(olGetSymbol(Program, KernelName, OL_SYMBOL_KIND_KERNEL, &Kernel));

  int *UsmPtr = nullptr;
  OL_CHECK(olMemAlloc(Gpu, OL_ALLOC_TYPE_HOST, sizeof(int),
                      reinterpret_cast<void **>(&UsmPtr)));
  *UsmPtr = -1;

  // liboffload has no buffers, allocate a USM pointer
  int *BufBacking = nullptr;
  OL_CHECK(olMemAlloc(Gpu, OL_ALLOC_TYPE_HOST, sizeof(int),
                      reinterpret_cast<void **>(&BufBacking)));
  *BufBacking = 5;


  ol_kernel_launch_size_args_t Launch{};
  Launch.Dimensions = 1;
  Launch.NumGroups = {1, 1, 1};
  Launch.GroupSize = {1, 1, 1};
  // Two local_accessor<int,1>(range<1>(64)) -> 2 * 64 * 4 = 512 bytes.
  Launch.DynSharedMemory = 512;

  uint32_t LocalOffset0 = 0; // local_mem
  int Val = 42;
  uint32_t LocalOffset1 = 256; // local_mem2
  void *BufAccPtr = BufBacking;
  uint64_t BufAccOffset = 0;
  void *UsmKernelPtr = UsmPtr;

  void *ArgPtrs[] = {
    &LocalOffset0, &Val, &LocalOffset1, &BufAccPtr, &BufAccOffset, &UsmKernelPtr,
  };
  size_t ArgSizes[] = {
    sizeof(LocalOffset0), sizeof(Val), sizeof(LocalOffset1),
    sizeof(BufAccPtr), sizeof(BufAccOffset), sizeof(UsmKernelPtr),
  };

  OL_CHECK(olLaunchKernelWithPtrArgs(Queue, Gpu, Kernel, ArgPtrs, ArgSizes, &Launch));
  OL_CHECK(olSyncQueue(Queue));

  int Expected = 5 + 42 + 43;
  std::printf("usm_ptr[0] = %d (expected %d)\n", *UsmPtr, Expected);

  OL_CHECK(olMemFree(UsmPtr));
  OL_CHECK(olMemFree(BufBacking));
  OL_CHECK(olDestroyProgram(Program));
  OL_CHECK(olDestroyQueue(Queue));
  OL_CHECK(olShutDown());
  return 0;
}
