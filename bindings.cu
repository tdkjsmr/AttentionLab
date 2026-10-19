// 只负责 Python 绑定；kernel 和启动代码放在各阶段自己的文件中。
#include <torch/extension.h>

void l1_forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& scores, const at::Tensor& probabilities,
    const at::Tensor& output, bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &l1_forward, "L1：QK、稳定 softmax、PV");
}
