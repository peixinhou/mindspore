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
#include <algorithm>

#include "minddata/dataset/kernels/ir/vision/resize_ir.h"

#include "minddata/dataset/kernels/image/resize_op.h"

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {

namespace vision {

// ResizeOperation
ResizeOperation::ResizeOperation(std::vector<int32_t> size, InterpolationMode interpolation)
    : size_(size), interpolation_(interpolation) {}

ResizeOperation::~ResizeOperation() = default;

std::string ResizeOperation::Name() const { return kResizeOperation; }

Status ResizeOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorSize("Resize", size_));
  return Status::OK();
}

std::shared_ptr<TensorOp> ResizeOperation::Build() {
  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  int32_t height = size_[0];
  int32_t width = 0;

  // User specified the width value.
  if (size_.size() == 2) {
    width = size_[1];
  }

  return std::make_shared<ResizeOp>(height, width, interpolation_);
}

Status ResizeOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["size"] = size_;
  args["interpolation"] = interpolation_;
  *out_json = args;
  return Status::OK();
}

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
