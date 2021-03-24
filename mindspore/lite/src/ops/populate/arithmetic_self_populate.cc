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

#include "src/ops/arithmetic_self.h"
#include "src/common/log_adapter.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
using mindspore::schema::PrimitiveType_Abs;
using mindspore::schema::PrimitiveType_AbsGrad;
using mindspore::schema::PrimitiveType_Ceil;
using mindspore::schema::PrimitiveType_Cos;
using mindspore::schema::PrimitiveType_Erf;
using mindspore::schema::PrimitiveType_Floor;
using mindspore::schema::PrimitiveType_Log;
using mindspore::schema::PrimitiveType_LogGrad;
using mindspore::schema::PrimitiveType_LogicalNot;
using mindspore::schema::PrimitiveType_Neg;
using mindspore::schema::PrimitiveType_NegGrad;
using mindspore::schema::PrimitiveType_Reciprocal;
using mindspore::schema::PrimitiveType_Round;
using mindspore::schema::PrimitiveType_Rsqrt;
using mindspore::schema::PrimitiveType_Sin;
using mindspore::schema::PrimitiveType_Sqrt;
using mindspore::schema::PrimitiveType_Square;

namespace mindspore {
namespace lite {
OpParameter *PopulateArithmeticSelf(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticSelfParameter *arithmetic_self_param =
    reinterpret_cast<ArithmeticSelfParameter *>(malloc(sizeof(ArithmeticSelfParameter)));
  if (arithmetic_self_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticSelfParameter failed.";
    return nullptr;
  }
  memset(arithmetic_self_param, 0, sizeof(ArithmeticSelfParameter));
  arithmetic_self_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(arithmetic_self_param);
}

REG_POPULATE(PrimitiveType_Abs, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_Cos, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_Sin, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_Log, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_Neg, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_NegGrad, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_LogGrad, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_AbsGrad, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_Sqrt, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_Square, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_Rsqrt, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_LogicalNot, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_Floor, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_Ceil, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_Round, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_Reciprocal, PopulateArithmeticSelf)
REG_POPULATE(PrimitiveType_Erf, PopulateArithmeticSelf)
}  // namespace lite
}  // namespace mindspore
