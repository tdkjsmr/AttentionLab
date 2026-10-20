// 只负责 Python 绑定；kernel 和启动代码放在各阶段自己的文件中。
#include <torch/extension.h>

#if defined(ATTENTION_LAB_L1)
void l1_forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& scores, const at::Tensor& probabilities,
    const at::Tensor& output, bool causal);
#elif defined(ATTENTION_LAB_L2)
void l2_forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& output, bool causal);
#elif defined(ATTENTION_LAB_L3)
void l3_forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& output, bool causal);
#elif defined(ATTENTION_LAB_L4)
void l4_forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& output, bool causal);
#else
#error "请选择本次编译的 CUDA 阶段"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#if defined(ATTENTION_LAB_L1)
    m.def("forward", &l1_forward, "L1：QK、稳定 softmax、PV");
#elif defined(ATTENTION_LAB_L2)
    m.def("forward", &l2_forward, "L2：融合分块与在线 softmax");
#elif defined(ATTENTION_LAB_L3)
    m.def("forward", &l3_forward, "L3：WMMA 分块与在线 softmax");
#elif defined(ATTENTION_LAB_L4)
    m.def("forward", &l4_forward, "L4：Q 块并行的 WMMA 在线 softmax");
#endif
}
