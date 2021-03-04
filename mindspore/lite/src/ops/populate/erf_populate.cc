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

#include "src/ops/erf.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateErfParameter(const mindspore::lite::PrimitiveC *primitive) {
  OpParameter *erf_parameter = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (erf_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc ErfParameter failed.";
    return nullptr;
  }
  memset(erf_parameter, 0, sizeof(OpParameter));
  erf_parameter->type_ = primitive->Type();
  return erf_parameter;
}

Registry ErfParameterRegistry(schema::PrimitiveType_Erf, PopulateErfParameter);
}  // namespace lite
}  // namespace mindspore
