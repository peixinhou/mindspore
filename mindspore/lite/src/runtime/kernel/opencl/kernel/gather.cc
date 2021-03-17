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
#include <string>
#include <algorithm>
#include <set>
#include <utility>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/gather.h"
#include "src/runtime/kernel/opencl/cl/gather.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Gather;

namespace mindspore::kernel {

int GatherOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 2) {
    MS_LOG(ERROR) << "GatherOpenCLKernel only supports 2 input Tensor but get " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "GatherOpenCLKernel only supports 1 output Tensor but get " << out_tensors_.size();
    return RET_ERROR;
  }

  int input_ndim = in_tensors_.front()->shape().size();
  if (input_ndim < 0 || input_ndim > 4) {
    MS_LOG(ERROR) << "GatherOpenCLKernel only supports 1-4D input Tensor but get " << input_ndim << "D.";
    return RET_ERROR;
  }

  TypeId data_type = in_tensors_.at(1)->data_type();
  if (data_type != kNumberTypeInt32 && data_type != kNumberTypeInt64 && data_type != kNumberTypeFloat32 &&
      data_type != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "GatherOpenCLKernel only supports Int32/Int64/Float32/Float16 indices Tensor.";
    return RET_ERROR;
  }

  auto *param = reinterpret_cast<GatherParameter *>(this->op_parameter_);
  if (param != nullptr) {
    axis_ = param->axis_;
    if (axis_ < 0) {
      axis_ += input_ndim;
    }
    if (input_ndim != C4NUM && axis_ != 0) {
      axis_ += C4NUM - input_ndim;
    }
    if (axis_ < 0 || axis_ >= input_ndim) {
      MS_LOG(ERROR) << "axis is invalid: axis=" << axis_ << ".";
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "GatherOpenCLKernel param is nullptr";
    return RET_ERROR;
  }

  int indices_ndim = in_tensors_.at(1)->shape().size();
  if (indices_ndim > 0) {
    int shape_0 = in_tensors_.at(1)->shape().at(0);
    if (indices_ndim > 1 && shape_0 != 1 && axis_ != 0) {
      MS_LOG(ERROR) << "GatherOpenCLKernel only supports 1D indices Tensor but get " << indices_ndim << "D.  or"
                    << "axis_ is not supported " << axis_;
      return RET_ERROR;
    }
  }

  return RET_OK;
}

void GatherOpenCLKernel::SetConstArgs() {
  auto input = GpuTensorInfo(in_tensors_.front());
  auto output = GpuTensorInfo(out_tensors_.front());
  int indices_num = in_tensors_.at(1)->ElementsNum();
  cl_int4 src_size = {static_cast<cl_int>(input.W), static_cast<cl_int>(input.H), static_cast<cl_int>(input.Slice),
                      static_cast<cl_int>(input.N)};
  cl_int4 dst_size = {static_cast<cl_int>(output.W), static_cast<cl_int>(output.H), static_cast<cl_int>(output.Slice),
                      static_cast<cl_int>(output.N)};
  int arg_cnt = 3;
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, src_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, dst_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, indices_num);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt, axis_);
}

void GatherOpenCLKernel::SetGlobalLocal() {
  auto output = GpuTensorInfo(out_tensors_.front());
  local_size_ = {1, 1, 1};
  global_size_ = {output.W, output.N * output.H, output.Slice};
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
}

int GatherOpenCLKernel::Prepare() {
  std::string kernel_name = "gather";
  if (in_tensors_.at(0)->IsConst()) {
    intensor0_is_tensor = false;
    int ret = InitWeights(0);
    if (ret != RET_OK) {
      return ret;
    }
  }
  if (in_tensors_.at(1)->IsConst()) {
    intensor1_is_tensor = false;
    int ret = InitWeights(1);
    if (ret != RET_OK) {
      return ret;
    }
  }
  // input data is buffer
  if (!intensor0_is_tensor) {
    kernel_name += "_buff";
  }
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::string program_name = "gather";
  ocl_runtime_->LoadSource(program_name, gather_source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, {}, out_tensors_[0]->data_type());
#endif
  SetGlobalLocal();
  SetConstArgs();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int GatherOpenCLKernel::InitWeights(int TensorIndex) {
  auto in_tensor = in_tensors_.at(TensorIndex);
  auto allocator = ocl_runtime_->GetAllocator();
  auto data_type = in_tensor->data_type();
  auto data = in_tensor->data_c();
  if (TensorIndex == 0) {
    auto in_tensor_num = GpuTensorInfo(in_tensor).ElementsC4Num;
    size_t size = GpuTensorInfo(in_tensor).N * GpuTensorInfo(in_tensor).H * GpuTensorInfo(in_tensor).W;
    size_t size_c = GpuTensorInfo(in_tensor).C;
    size_t size_c4 = UP_DIV(size_c, C4NUM) * C4NUM;
    if (data_type == kNumberTypeFloat32) {
      data_ = reinterpret_cast<float_t *>(allocator->Malloc(sizeof(float_t) * in_tensor_num));
      memset(data_, 0x00, in_tensor_num);
      allocator->MapBuffer(data_, CL_MAP_WRITE, nullptr, true);
      for (int i = 0; i < size; ++i) {
        memcpy(reinterpret_cast<float_t *>(data_) + i * size_c4, reinterpret_cast<float_t *>(data) + i * size_c,
               sizeof(float_t) * size_c);
      }
    } else if (data_type == kNumberTypeFloat16) {
      data_ = reinterpret_cast<float16_t *>(allocator->Malloc(sizeof(float16_t) * in_tensor_num));
      memset(data_, 0x00, in_tensor_num);
      allocator->MapBuffer(data_, CL_MAP_WRITE, nullptr, true);
      for (int i = 0; i < size; ++i) {
        memcpy(reinterpret_cast<float16_t *>(data_) + i * size_c4, reinterpret_cast<float16_t *>(data) + i * size_c,
               sizeof(float16_t) * size_c);
      }
    }
    allocator->UnmapBuffer(data_);
  } else {
    auto in_tensor_num = GpuTensorInfo(in_tensor).ElementsNum;
    indices_data_ = reinterpret_cast<int32_t *>(allocator->Malloc(sizeof(int32_t) * in_tensor_num));
    allocator->MapBuffer(indices_data_, CL_MAP_WRITE, nullptr, true);
    memcpy(indices_data_, data, sizeof(data_type) * in_tensor_num);
    allocator->UnmapBuffer(indices_data_);
  }
  if (indices_data_ == nullptr && data_ == nullptr) {
    MS_LOG(ERROR) << "InitWeights Memory allocation failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int GatherOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  auto input_0_ptr = intensor0_is_tensor ? in_tensors_.front()->data_c() : data_;
  auto input_1_ptr = intensor1_is_tensor ? in_tensors_.at(1)->data_c() : indices_data_;
  ocl_runtime_->SetKernelArg(kernel_, 0, out_tensors_.front()->data_c());
  ocl_runtime_->SetKernelArg(kernel_, 1, input_0_ptr);
  ocl_runtime_->SetKernelArg(kernel_, 2, input_1_ptr, lite::opencl::MemType::BUF);
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Gather, OpenCLKernelCreator<GatherOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Gather, OpenCLKernelCreator<GatherOpenCLKernel>);

}  // namespace mindspore::kernel
