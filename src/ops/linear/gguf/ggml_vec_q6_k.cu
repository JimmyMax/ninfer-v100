// The vector products for one block type, in their own unit so they build in parallel with the others.
#include "ggml_bridge_vec.cuh"

NINFER_GGUF_VECTOR_INSTANCE(GGML_TYPE_Q6_K);
// 本仓库 V100 增量：Q6_K 密集逐行布局的向量路线与 slice 反量化。
NINFER_GGUF_Q6K_DENSE_INSTANCE;
