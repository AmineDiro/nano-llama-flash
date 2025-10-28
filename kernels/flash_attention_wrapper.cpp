#include <torch/extension.h>

torch::Tensor flash_attn_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    const int Bc
    );

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_forward", torch::wrap_pybind_function(flash_attn_forward), "flash_attn_forward");
}
