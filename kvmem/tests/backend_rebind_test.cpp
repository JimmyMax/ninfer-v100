#include "kvmem/kvmem_runtime.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>

using namespace kvmem;

static int g_fail = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

struct RecordingBackend : KvMemBackend {
    int32_t next = 0;
    std::vector<int32_t> allocs;
    std::vector<int32_t> frees;
    std::vector<std::string> ops;

    int32_t alloc_gpu_slot() override {
        const int32_t s = next++;
        allocs.push_back(s);
        ops.push_back("alloc");
        return s;
    }
    void free_gpu_slot(int32_t slot) override {
        frees.push_back(slot);
        ops.push_back("free");
    }
};

static KvMemRuntimeConfig make_cfg() {
    KvMemRuntimeConfig cfg;
    cfg.store.block_tokens = 32;
    cfg.store.select_budget = 32 * 4;
    cfg.store.sink_blocks = 1;
    cfg.store.recent_blocks = 1;
    cfg.store.estimated_block_bytes = 1024;
    cfg.cpu_bytes = 1024 * 16;
    return cfg;
}

static void test_backend_rebind() {
    RecordingBackend first, second;
    KvMemRuntime rt(make_cfg(), &first);
    rt.register_append(64);
    for (uint32_t id = 0; id < 2; ++id) rt.store().set_block_gpu_slot(id, first.alloc_gpu_slot());
    bool refused = false;
    try { rt.rebind_backend(&second); } catch (const std::logic_error &) { refused = true; }
    CHECK(refused);
    rt.prepare_selection({});
    refused = false;
    try { rt.rebind_backend(&second); } catch (const std::logic_error &) { refused = true; }
    CHECK(refused);
    rt.finish_reselect();
    CHECK(first.frees.size() == 2);
    rt.rebind_backend(nullptr);
    rt.rebind_backend(&second);
    rt.prepare_selection({0, 1});
    rt.finish_reselect();
    CHECK(second.allocs.size() == 2);
    CHECK(first.allocs.size() == 2);
    rt.truncate_to(0);
    CHECK(second.frees.size() == 2);
}

int main() {
    test_backend_rebind();
    return g_fail ? 1 : 0;
}
