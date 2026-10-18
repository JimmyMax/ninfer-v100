// The vendored ggml CUDA headers (third_party/ggml-quants/src/ggml-cuda/common.cuh) declare two
// host helpers without shipping the common.cu that defines them, so we define both in
// src/ops/linear/gguf/ggml_bridge.cu. A Release build links either way: -O3 drops the unused inline
// callers (CUDA_CHECK_GEN and the mmq/fattn headers) that would emit the references. A build that
// keeps them -- Debug, -O0, -lineinfo -- failed with "undefined reference to ggml_cuda_get_device()"
// and "undefined reference to ggml_cuda_error(...)" until those definitions existed.
//
// These addresses are in a static initializer, so the linker must resolve both symbols for this
// test to build at all, at any optimisation level. Deleting the definitions from the bridge turns
// into a link failure here instead of a report from someone building a debug binary.

#include <cstdio>

// Declared exactly as common.cuh declares them.
[[noreturn]] void ggml_cuda_error(const char* stmt, const char* func, const char* file, int line, const char* msg);
int ggml_cuda_get_device();

namespace {

void (*volatile error_helper)(const char*, const char*, const char*, int, const char*) = &ggml_cuda_error;
int (*volatile device_helper)()                                                         = &ggml_cuda_get_device;

} // namespace

int main() {
    // Never called: both helpers abort on purpose. Printing the pointers keeps the references live
    // in the generated code, which is the whole point of the test.
    std::printf("vendor ggml CUDA host helpers resolve: %p, %p\n",
                reinterpret_cast<const void*>(error_helper), reinterpret_cast<const void*>(device_helper));
    return 0;
}
