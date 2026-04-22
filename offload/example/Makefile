LLVM_INSTALL  ?=
CUDA_PATH     ?= /usr/local/cuda

SM_ARCH       ?= sm_75
PTX_VERSION   ?= 8.0

KERNEL_NAME   ?= _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_7nd_itemILi1EEEE_

HOST_BIN      := kernel_args_offload
HOST_SYCL_BIN := kernel_args_sycl

SPV_ASM       := kernel_args_verify.spvasm
SPV_BIN       := kernel_args_verify.spv

PTX_SRC       := kernel_args_verify.ptx
CUBIN         := kernel_args_verify.cubin
FATBIN        := kernel_args_verify.fatbin

all: $(HOST_BIN)

$(HOST_BIN): kernel_args_offload.cpp
	clang++ -I$(LLVM_INSTALL)/include/ -L$(LLVM_INSTALL)/lib -Wl,-rpath,$(LLVM_INSTALL)/lib -lLLVMOffload \
	$< $(LDFLAGS) $(LDLIBS) -o $@

# for cuda add -fsycl-targets=nvptx64-nvidia-cuda
$(HOST_SYCL_BIN): kernel_args_sycl.cpp
	clang -fsycl-targets=nvptx64-nvidia-cuda -fsycl-device-code-split=off -fsycl kernel_args_sycl.cpp -o $@

sycl: $(HOST_SYCL_BIN)

spv: $(SPV_BIN)

$(SPV_BIN): $(SPV_ASM)
	spirv-as --target-env spv1.4 $< -o $@

ptx: $(FATBIN)

$(CUBIN): $(PTX_SRC)
	ptxas -arch=$(SM_ARCH) -o $@ $<

$(FATBIN): $(CUBIN) $(PTX_SRC)
	fatbinary --64 --create=$@ --image=profile=$(SM_ARCH),file=$(CUBIN) \
	--image=profile=compute_$(patsubst sm_%,%,$(SM_ARCH)),file=$(PTX_SRC)

run-cuda: $(HOST_BIN) $(FATBIN)
	./$(HOST_BIN) $(FATBIN) $(KERNEL_NAME)

run-spirv: $(HOST_BIN) $(SPV_BIN)
	./$(HOST_BIN) $(SPV_BIN) $(KERNEL_NAME)

run-sycl: $(HOST_SYCL_BIN)
	./$(HOST_SYCL_BIN)

clean:
	rm -f $(HOST_BIN) $(HOST_SYCL_BIN) $(SPV_BIN) $(CUBIN) $(FATBIN)
