/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/populate/arithmetic_populate.h"
#include "src/ops/arithmetic.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
using mindspore::schema::PrimitiveType_Equal;
using mindspore::schema::PrimitiveType_FloorDiv;
using mindspore::schema::PrimitiveType_FloorMod;
using mindspore::schema::PrimitiveType_Greater;
using mindspore::schema::PrimitiveType_GreaterEqual;
using mindspore::schema::PrimitiveType_Less;
using mindspore::schema::PrimitiveType_LessEqual;
using mindspore::schema::PrimitiveType_LogicalAnd;
using mindspore::schema::PrimitiveType_LogicalOr;
using mindspore::schema::PrimitiveType_Maximum;
using mindspore::schema::PrimitiveType_Minimum;
using mindspore::schema::PrimitiveType_Mod;
using mindspore::schema::PrimitiveType_NotEqual;
using mindspore::schema::PrimitiveType_RealDiv;
using mindspore::schema::PrimitiveType_SquaredDifference;

namespace mindspore {
namespace lite {

ArithmeticParameter *PopulateArithmeticCommonPara(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ArithmeticParameter));
  param->op_parameter_.type_ = primitive->Type();
  param->broadcasting_ = reinterpret_cast<const lite::Arithmetic *>(primitive)->Broadcasting();
  param->ndim_ = reinterpret_cast<const lite::Arithmetic *>(primitive)->NDims();
  param->activation_type_ = 0;

  auto tmp_shape = reinterpret_cast<const lite::Arithmetic *>(primitive)->InShape0();
  memcpy(param->in_shape0_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  tmp_shape = reinterpret_cast<const lite::Arithmetic *>(primitive)->InShape1();
  memcpy(param->in_shape1_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  tmp_shape = reinterpret_cast<const lite::Arithmetic *>(primitive)->OutputShape();
  memcpy(param->out_shape_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  return param;
}

OpParameter *PopulateArithmetic(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *param = PopulateArithmeticCommonPara(primitive);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_RealDiv, PopulateArithmetic)
REG_POPULATE(PrimitiveType_LogicalAnd, PopulateArithmetic)
REG_POPULATE(PrimitiveType_LogicalOr, PopulateArithmetic)
REG_POPULATE(PrimitiveType_Equal, PopulateArithmetic)
REG_POPULATE(PrimitiveType_NotEqual, PopulateArithmetic)
REG_POPULATE(PrimitiveType_Less, PopulateArithmetic)
REG_POPULATE(PrimitiveType_LessEqual, PopulateArithmetic)
REG_POPULATE(PrimitiveType_Greater, PopulateArithmetic)
REG_POPULATE(PrimitiveType_GreaterEqual, PopulateArithmetic)
REG_POPULATE(PrimitiveType_Maximum, PopulateArithmetic)
REG_POPULATE(PrimitiveType_Minimum, PopulateArithmetic)
REG_POPULATE(PrimitiveType_FloorDiv, PopulateArithmetic)
REG_POPULATE(PrimitiveType_FloorMod, PopulateArithmetic)
REG_POPULATE(PrimitiveType_Mod, PopulateArithmetic)
REG_POPULATE(PrimitiveType_SquaredDifference, PopulateArithmetic)
}  // namespace lite
}  // namespace mindspore
