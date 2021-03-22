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

#include "nnacl/fp32/gelu_fp32.h"
#include "nnacl/errorcode.h"

int Gelu(const float *src, float *dst, int64_t length, bool approximate) {
  if (src == NULL || dst == NULL) {
    return NNACL_ERR;
  }
  int i = 0;
  if (approximate) {
    // dst = 0.5 * x * (1 + tanh((2 / pi) ^ 0.5 * (x + 0.044715x^3)))
#if defined(ENABLE_AVX)
    int C8 = UP_ROUND(length, C8NUM);
    for (; i < C8; i += C8NUM) {
      MS_FLOAT32X8 in = MS_LD256_F32(src + i);
      MS_FLOAT32X8 res = 0.5 * in * (1.0 + MS_TANHX8_F32((0.79788456080287f + 0.035677408136f * in * in) * in));
      MS_ST256_F32(dst + i, res);
    }
#endif
#if defined(ENABLE_SSE) || defined(ENABLE_ARM)
    int C4 = UP_ROUND(length, C4NUM);
    for (; i < C4; i += C4NUM) {
      MS_FLOAT32X4 in = MS_LDQ_F32(src + i);
      MS_FLOAT32X4 res = 0.5 * in * (1.0 + MS_TANHX4_F32((0.79788456080287f + 0.035677408136f * in * in) * in));
      MS_STQ_F32(dst + i, res);
    }
#endif
    for (; i < length; i++) {
      dst[i] = 0.5f * src[i] * (1.0 + TanhOpt((0.79788456080287f + 0.035677408136f * src[i] * src[i]) * src[i]));
    }
  } else {
#if defined(ENABLE_AVX) || defined(ENABLE_SSE) || defined(ENABLE_ARM)
    int C4 = UP_ROUND(length, C4NUM);
    for (; i < C4; i += C4NUM) {
      MS_FLOAT32X4 in = MS_LDQ_F32(src + i);
      MS_FLOAT32X4 res = 0.5 * in * (1.0 + MS_ERFX4_F32(in / 1.4142135623730951f));
      MS_STQ_F32(dst + i, res);
    }
#endif
    for (; i < length; i++) {
      dst[i] = 0.5f * src[i] * (1.0f + erff(src[i] / 1.4142135623730951f));
    }
  }
  return NNACL_OK;
}
