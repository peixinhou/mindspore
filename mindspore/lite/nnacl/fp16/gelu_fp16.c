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

#include "nnacl/fp16/gelu_fp16.h"
#include "nnacl/errorcode.h"

int GeluFp16(const float16_t *src, float16_t *dst, int64_t length, bool approximate) {
  if (src == NULL || dst == NULL) {
    return NNACL_ERR;
  }
  int i = 0;
  if (approximate) {
    // dst = 0.5 * x * (1 + tanh((2 / pi) ^ 0.5 * (x + 0.044715x^3)))
#ifdef ENABLE_NEON
    int C8 = UP_ROUND(length, C8NUM);
    for (; i < C8; i += C8NUM) {
      float16x8_t in = vld1q_f16(src + i);
      float16x8_t res =
        0.5 * in * (1.0 + MS_TANHX8_F16(((float16_t)0.79788456080287 + (float16_t)0.035677408136 * in * in) * in));
      vst1q_f16(dst + i, res);
    }
#endif
    for (; i < length; i++) {
      dst[i] =
        0.5 * src[i] *
        (1.0 + TanhOptFp16(((float16_t)0.79788456080287 + (float16_t)0.035677408136 * src[i] * src[i]) * src[i]));
    }
  } else {
#ifdef ENABLE_NEON
    int C8 = UP_ROUND(length, C8NUM);
    for (; i < C8; i += C8NUM) {
      float16x8_t in = vld1q_f16(src + i);
      float16x8_t res = 0.5 * in * (1.0 + MS_ERFX8_F16(in / (float16_t)1.4142135623730951f));
      vst1q_f16(dst + i, res);
    }
#endif
    for (; i < length; i++) {
      dst[i] = 0.5 * src[i] * (1.0 + erff(src[i] / 1.4142135623730951f));
    }
  }
  return NNACL_OK;
}
