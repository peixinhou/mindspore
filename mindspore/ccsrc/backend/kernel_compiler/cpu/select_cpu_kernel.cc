/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "backend/kernel_compiler/cpu/select_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void SelectCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 3) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but SelectCpuKernel needs 3 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but SelectCpuKernel needs 1 output.";
  }
  auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  for (size_t x : shape) {
    element_num_ *= x;
  }
  return;
}

template <typename T>
bool SelectCPUKernel<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                const std::vector<AddressPtr> &outputs) {
  auto input_cond = reinterpret_cast<bool *>(inputs[0]->addr);
  auto input_x = reinterpret_cast<T *>(inputs[1]->addr);
  auto input_y = reinterpret_cast<T *>(inputs[2]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  auto task = [=](const size_t start, const size_t end) {
    for (size_t pos = start; pos < end; pos++) {
      output[pos] = input_cond[pos] ? input_x[pos] : input_y[pos];
    }
  };
  CPUKernelUtils::ParallelFor(task, element_num_);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
