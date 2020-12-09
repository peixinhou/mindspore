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

#include "tools/converter/parser/onnx/onnx_matmul_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
lite::PrimitiveC *OnnxMatmulParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                       const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx MatMulParser";
  auto attr = std::make_unique<schema::MatMulT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  float alpha = 1.0f;
  float beta = 1.0f;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "transA") {
      attr->transposeA = static_cast<bool>(onnx_node_attr.i());
    } else if (attribute_name == "transB") {
      attr->transposeB = static_cast<bool>(onnx_node_attr.i());
    } else if (attribute_name == "alpha") {
      alpha = onnx_node_attr.f();
    } else if (attribute_name == "beta") {
      beta = onnx_node_attr.f();
    }
  }
  if (alpha != 1 || beta != 1) {
    MS_LOG(ERROR) << "not support alpha * A * B + beta * C";
    return nullptr;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_MatMul;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

OnnxNodeRegistrar g_onnxMatmulParser("MatMul", new OnnxMatmulParser());
}  // namespace lite
}  // namespace mindspore
