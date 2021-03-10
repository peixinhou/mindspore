/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <stdlib.h>
#include "src/runtime/kernel/arm/fp32/uniform_real_fp32.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_UniformReal;

namespace mindspore::kernel {
int UniformRealCPUKernel::Init() { return RET_OK; }

int UniformRealCPUKernel::ReSize() { return RET_OK; }

int UniformRealCPUKernel::Run() {
  auto output0 = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  MS_ASSERT(output0);
  if (seed_ < 0 || seed2_ < 0) {
    MS_LOG(ERROR) << "seed_:" << seed_ << " and seed2_:" << seed2_ << " must be greater than 0!";
    return RET_ERROR;
  }
  std::srand(seed_ || seed2_);
  for (int i = 0; i < out_tensors_.at(0)->ElementsNum(); ++i) {
    output0[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_UniformReal, LiteKernelCreator<UniformRealCPUKernel>)
}  // namespace mindspore::kernel
