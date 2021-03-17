/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <cstring>
#include <algorithm>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/arithmetic_self.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/arithmeticself.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

int ArithmeticSelfOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 1 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (!IsArithmeticSelf(Type()) && this->op_parameter_->type_ != schema::PrimitiveType_Erf &&
      this->op_parameter_->type_ != schema::PrimitiveType_Reciprocal) {
    MS_LOG(ERROR) << "UnSupported Operator: " << schema::EnumNamePrimitiveType(Type());
    return RET_ERROR;
  }
  if (in_tensors_.at(0)->shape() != out_tensors_.at(0)->shape()) {
    MS_LOG(ERROR) << "input shape != output shape ";
    return RET_ERROR;
  }
  return RET_OK;
}

void ArithmeticSelfGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
  const int max_divider = 8;
  const int max_x = 4, max_y = 8;
  size_t x = std::min(GetMaxDivisorStrategy1(global[0], max_divider), max_x);
  size_t yz = max_size / x;
  size_t y = std::min<size_t>(std::min<size_t>(GetMaxDivisorStrategy1(global[1], max_divider), yz), max_y);
  size_t z = std::min<size_t>(yz / y, static_cast<int>(UP_DIV(global[2], 2)));
  *local = {x, y, z};
}

void ArithmeticSelfOpenCLKernel::SetGlobalLocal() {
  const std::vector<size_t> &max_global = ocl_runtime_->GetWorkItemSize();
  global_size_ = {output_shape_.N * output_shape_.H, output_shape_.W, output_shape_.Slice};
  ArithmeticSelfGetWorkGroup(global_size_, &local_size_, max_global[0]);
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
}

void ArithmeticSelfOpenCLKernel::SetConstArgs() {
  cl_int4 output_shape = {static_cast<cl_int>(output_shape_.N), static_cast<cl_int>(output_shape_.H),
                          static_cast<cl_int>(output_shape_.W), static_cast<cl_int>(output_shape_.Slice)};
  ocl_runtime_->SetKernelArg(kernel_, 2, output_shape);
}

int ArithmeticSelfOpenCLKernel::Prepare() {
  output_shape_ = GpuTensorInfo(out_tensors_[0]);
  std::string kernel_name = schema::EnumNamePrimitiveType(Type());
  std::string program_name = "ArithmeticSelf";
  ocl_runtime_->LoadSource(program_name, arithmeticself_source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
  SetGlobalLocal();
  SetConstArgs();
  return RET_OK;
}

int ArithmeticSelfOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_.front()->data_c());
  ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_.front()->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Abs, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Ceil, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Cos, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Exp, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Floor, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Log, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_LogicalNot, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Round, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Rsqrt, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Sin, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Neg, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Sqrt, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Square, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Erf, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Reciprocal, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Abs, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Ceil, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Cos, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Exp, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Floor, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Log, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_LogicalNot, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Round, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Rsqrt, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Sin, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Neg, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Sqrt, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Square, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Erf, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Reciprocal, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)

}  // namespace mindspore::kernel
