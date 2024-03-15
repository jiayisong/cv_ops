// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void grid_sample_deterministic_forward_cuda(Tensor  input,Tensor  offset,Tensor  output, const int offset_num, const int channel,  const int height, const int width);
void grid_sample_deterministic_input_backward_cuda(Tensor grad_output, Tensor t2_indice, Tensor offset, Tensor batch_id, Tensor t2_i_i_s_id, Tensor t2_i_i_f_id, Tensor grad_input, const int group, const int offset_num, const int channel, const int height, const int width);
void grid_sample_deterministic_offset_backward_cuda(Tensor grad_output, Tensor input, Tensor offset, Tensor grad_offset, const int offset_num, const int channel,  const int height, const int width);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("grid_sample_deterministic_forward", &grid_sample_deterministic_forward_cuda, "grid_sample_deterministic_forward");
	m.def("grid_sample_deterministic_input_backward", &grid_sample_deterministic_input_backward_cuda, "grid_sample_deterministic_input_backward");
	m.def("grid_sample_deterministic_offset_backward", &grid_sample_deterministic_offset_backward_cuda, "grid_sample_deterministic_offset_backward");
}
