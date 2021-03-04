/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <cmath>
#include <vector>
#include "src/runtime/kernel/arm/base/erf.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Erf;

namespace mindspore::kernel {
int ErfCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ErfCPUKernel::Run() {
  auto input_data = reinterpret_cast<float *>(in_tensors_.front()->data_c());
  auto output_data = reinterpret_cast<float *>(out_tensors_.front()->data_c());
  int data_ele = in_tensors_.front()->ElementsNum();
  for (int i = 0; i < data_ele; ++i) {
    output_data[i] = std::erf(input_data[i]);
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Erf, LiteKernelCreator<ErfCPUKernel>)
}  // namespace mindspore::kernel
