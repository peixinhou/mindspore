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

#include "src/runtime/kernel/npu/pad_npu.h"
#include <memory>
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"
using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Pad;

namespace mindspore::kernel {
int PadNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                            OpParameter *opParameter) {
  if (inputs.size() == 2) {
    if (inputs[1]->data_c() == nullptr && inputs[1]->ElementsNum() != 8) {
      MS_LOG(ERROR) << "pad input 2 nullptr or paddings size " << inputs[1]->ElementsNum() << " unsupported";
      return RET_ERROR;
    }
    for (int i = 0; i < inputs[1]->ElementsNum(); i++) {
      param_->paddings_[i] = static_cast<int *>(inputs[1]->data_c())[i];
    }
  } else if (inputs.size() == 1) {
    if (pad_->GetPaddings().size() != 8) {
      MS_LOG(WARNING) << "NPU only support paddings size 8";
      return RET_ERROR;
    }
    for (int i = 0; i < pad_->GetPaddings().size(); i++) {
      param_->paddings_[i] = pad_->GetPaddings()[i];
    }
  } else {
    MS_LOG(ERROR) << "pad input size " << inputs.size() << " not supported";
    return RET_ERROR;
  }
  if (pad_->GetPaddingMode() != schema::PaddingMode_CONSTANT &&
      pad_->GetPaddingMode() != schema::PaddingMode_SYMMETRIC &&
      pad_->GetPaddingMode() != schema::PaddingMode_REFLECT) {
    MS_LOG(ERROR) << "pad npu not support mode " << pad_->GetPaddingMode();
    return RET_ERROR;
  }
  return RET_OK;
}

int PadNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                               const std::vector<ge::Operator *> &npu_inputs) {
  ge::TensorDesc padding_tensor_desc(ge::Shape({4, 2}), ge::FORMAT_NCHW, ge::DT_INT32);
  ge::TensorPtr padding_tensor = std::make_shared<hiai::Tensor>(padding_tensor_desc);
  padding_tensor->SetData(reinterpret_cast<uint8_t *>(param_->paddings_), 8 * sizeof(int));
  paddings_ = new hiai::op::Const(name_ + "paddings");
  paddings_->set_attr_value(padding_tensor);
  if (pad_->GetPaddingMode() == schema::PaddingMode_CONSTANT) {
    op_ = new (std::nothrow) hiai::op::PadV2(name_);
    if (op_ == nullptr) {
      MS_LOG(ERROR) << name_ << " op is nullptr";
      return RET_ERROR;
    }

    ge::TensorDesc constant_values_tensor_desc(ge::Shape({1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorPtr constant_values_tensor = std::make_shared<hiai::Tensor>(constant_values_tensor_desc);
    vector<float> constant_values_data_value = {pad_->GetConstantValue()};
    constant_values_tensor->SetData(reinterpret_cast<uint8_t *>(constant_values_data_value.data()), 1 * sizeof(float));
    constant_ = new hiai::op::Const(name_ + "constant");
    constant_->set_attr_value(constant_values_tensor);

    op_->set_input_x(*npu_inputs[0]);
    op_->set_input_constant_values(*constant_);
    op_->set_input_paddings(*paddings_);
  } else {
    mirror_op_ = new (std::nothrow) hiai::op::MirrorPad(name_);
    if (mirror_op_ == nullptr) {
      MS_LOG(ERROR) << name_ << " op is nullptr";
      return RET_ERROR;
    }
    mirror_op_->set_input_x(*npu_inputs[0]);
    mirror_op_->set_input_paddings(*paddings_);
    if (pad_->GetPaddingMode() == schema::PaddingMode_SYMMETRIC) {
      mirror_op_->set_attr_mode("SYMMETRIC");
    } else {
      mirror_op_->set_attr_mode("REFLECT");
    }
  }

  return RET_OK;
}

ge::Operator *mindspore::kernel::PadNPUKernel::GetNPUOp() {
  if (pad_->GetPaddingMode() == schema::PaddingMode_CONSTANT) {
    return op_;
  }
  return mirror_op_;
}

PadNPUKernel::~PadNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
  if (mirror_op_ != nullptr) {
    delete mirror_op_;
    mirror_op_ = nullptr;
  }
  if (paddings_ != nullptr) {
    delete paddings_;
    paddings_ = nullptr;
  }
  if (constant_ != nullptr) {
    delete constant_;
    constant_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Pad, NPUKernelCreator<PadNPUKernel>)
}  // namespace mindspore::kernel
