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

#ifndef MINDSPORE_NNACL_FP32_CONV_COMMON_H_
#define MINDSPORE_NNACL_FP32_CONV_COMMON_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/pack.h"
#include "nnacl/op_base.h"
#include "nnacl/common_func.h"
#include "nnacl/conv_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

// fp32 convolution common (im2col+gemm)
void ConvFp32(const float *input_data, float *packed_input, const float *packed_weight, const float *bias_data,
              float *col_major_input, float *output_data, int task_id, const ConvParameter *conv_param);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_CONV_COMMON_H_
