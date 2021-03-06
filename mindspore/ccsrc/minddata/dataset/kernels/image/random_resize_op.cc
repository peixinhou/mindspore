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
#include "minddata/dataset/kernels/image/random_resize_op.h"

#include <random>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const int32_t RandomResizeOp::kDefTargetWidth = 0;

Status RandomResizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  // Randomly selects from the following four interpolation methods
  // 0-bilinear, 1-nearest_neighbor, 2-bicubic, 3-area
  interpolation_ = static_cast<InterpolationMode>(distribution_(random_generator_));
  return ResizeOp::Compute(input, output);
}
}  // namespace dataset
}  // namespace mindspore
