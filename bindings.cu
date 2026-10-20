// 只负责 Python 绑定；kernel 和启动代码放在各阶段自己的文件中。
#include <torch/extension.h>

#if defined(ATTENTION_LAB_L4) || defined(ATTENTION_LAB_L5)
#include <pybind11/stl.h>
#include "kernels/launch_resources.cuh"
#endif

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
LaunchResources l4_resources(const at::Tensor& q);
void l4_forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& output, bool causal);
#elif defined(ATTENTION_LAB_L5)
LaunchResources l5_resources(const at::Tensor& q);
void l5_configure(const at::Tensor& q);
void l5_forward(
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
    m.def("resources", &l4_resources, "L4：编译资源与理论驻留上限");
    m.def("forward", &l4_forward, "L4：Q 块并行的 WMMA 在线 softmax");
#elif defined(ATTENTION_LAB_L5)
    m.def("resources", &l5_resources, "L5：编译资源与理论驻留上限");
    m.def("configure", &l5_configure, "L5：配置动态 shared 上限");
    m.def("forward", &l5_forward, "L5：异步拷贝与 WMMA 在线 softmax");
#endif
}
