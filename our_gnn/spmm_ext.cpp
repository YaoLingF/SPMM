#include <torch/extension.h>

torch::Tensor our_spmm_forward_cuda(torch::Tensor rowptr,
                                    torch::Tensor col,
                                    torch::Tensor val,
                                    torch::Tensor x);

torch::Tensor forward(torch::Tensor rowptr,
                      torch::Tensor col,
                      torch::Tensor val,
                      torch::Tensor x) {
  TORCH_CHECK(rowptr.is_cuda(), "rowptr must be a CUDA tensor");
  TORCH_CHECK(col.is_cuda(), "col must be a CUDA tensor");
  TORCH_CHECK(val.is_cuda(), "val must be a CUDA tensor");
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  return our_spmm_forward_cuda(rowptr, col, val, x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "CSR SpMM forward (CUDA)");
}
