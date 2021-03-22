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
#ifndef MINDSPORE_LITE_NNACL_FP16_GELU_H_
#define MINDSPORE_LITE_NNACL_FP16_GELU_H_

#include "nnacl/op_base.h"
#include "nnacl/fp16/activation_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif
int GeluFp16(const float16_t *src, float16_t *out, int64_t real_dst_count, bool approximate);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP16_GELU_H_
