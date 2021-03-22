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

#include "src/runtime/kernel/arm/fp16/gelu_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "nnacl/fp16/gelu_fp16.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_GeLU;

namespace mindspore::kernel {
int GeluFp16CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int GeluFp16CPUKernel::ReSize() { return RET_OK; }

int GeluFp16CPUKernel::DoGelu(int task_id) {
  auto size = in_tensors_.at(0)->ElementsNum();
  int stride = UP_DIV(size, thread_count_);
  int len = MSMIN(stride, size - stride * task_id);
  if (len <= 0) {
    return RET_OK;
  }
  auto ret = GeluFp16(in_data_ + stride * task_id, out_data_ + stride * task_id, len, approximate_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoGelu in task_id:" << task_id << " is error!";
  }
  return ret;
}

int GeluRunFp16(void *cdata, int task_id) {
  auto gelu_kernel = reinterpret_cast<GeluFp16CPUKernel *>(cdata);
  auto error_code = gelu_kernel->DoGelu(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "GeluRun error task_id[" << task_id << "] error_code[" << error_code << "]";
  }
  return error_code;
}

int GeluFp16CPUKernel::Run() {
  in_data_ = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data_c());
  MS_ASSERT(in_data_ != nullptr);
  out_data_ = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data_c());
  MS_ASSERT(out_data_ != nullptr);
  auto ret = ParallelLaunch(this->context_->thread_pool_, GeluRunFp16, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Gelu fp16 function error error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_GeLU, LiteKernelCreator<GeluFp16CPUKernel>)
}  // namespace mindspore::kernel
