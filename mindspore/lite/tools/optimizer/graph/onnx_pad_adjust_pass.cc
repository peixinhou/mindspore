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
#include "tools/optimizer/graph/onnx_pad_adjust_pass.h"
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include "mindspore/lite/include/errorcode.h"
#include "src/ops/primitive_c.h"
#include "src/param_value_lite.h"

namespace mindspore::opt {
ValueNodePtr CreateNewValueNode(void *attr, const schema::PrimitiveType &op_type) {
  if (attr == nullptr) {
    MS_LOG(ERROR) << "attr is nullptr";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = op_type;
  primitive->value.value = attr;
  auto primitive_c = lite::PrimitiveC::Create(primitive.release());
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "create primitivec nullptr";
    return nullptr;
  }
  return NewValueNode(std::shared_ptr<lite::PrimitiveC>(primitive_c));
}
ParameterPtr CreateNewParameter(const FuncGraphPtr &func_graph, const std::vector<int> &data) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(data != nullptr);
  auto parameter = func_graph->add_parameter();
  std::vector<int> shape;
  shape.push_back(static_cast<int>(data.size()));
  std::vector<int64_t> shape_vector;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                       [](const int32_t &value) { return static_cast<int64_t>(value); });
  auto type_id = static_cast<TypeId>(kNumberTypeInt32);
  auto type_ptr = TypeIdToType(type_id);
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  parameter->set_abstract(abstract_tensor);

  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  MS_ASSERT(param_value != nullptr);
  param_value->set_tensor_shape(shape);
  param_value->set_tensor_type(type_id);
  param_value->set_format(schema::Format_NCHW);

  size_t size = data.size() * sizeof(int);
  auto tensor_data = new (std::nothrow) uint8_t[size];
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "tensor_data is nullptr";
    return nullptr;
  }
  auto ret = memcpy_s(tensor_data, size, data.data(), size);
  if (ret != 0) {
    MS_LOG(ERROR) << "set tensor data failed.";
    return nullptr;
  }
  param_value->SetTensorData(tensor_data, size);
  parameter->set_default_param(param_value);
  return parameter;
}

CNodePtr NewReshapeOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr input_node, const std::vector<int> &shape) {
  auto attr = std::make_unique<schema::ReshapeT>();
  attr->format = schema::Format_NCHW;
  for (auto ele : shape) {
    attr->shape.push_back(ele);
  }
  ValueNodePtr value_node = CreateNewValueNode(attr.release(), schema::PrimitiveType_Reshape);

  auto new_parameter = CreateNewParameter(func_graph, shape);
  new_parameter->set_name(input_node->fullname_with_scope() + "_reshape/shape");
  std::vector<AnfNodePtr> op_inputs = {value_node, input_node, new_parameter};
  auto reshape = func_graph->NewCNode(op_inputs);
  reshape->set_fullname_with_scope(input_node->fullname_with_scope() + "_reshape");
  return reshape;
}

CNodePtr NewTransposeOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr input_node, std::vector<int> perm) {
  auto attr = std::make_unique<schema::TransposeT>();
  for (auto ele : perm) {
    attr->perm.push_back(ele);
  }
  ValueNodePtr value_node = CreateNewValueNode(attr.release(), schema::PrimitiveType_Transpose);

  std::vector<AnfNodePtr> op_inputs = {value_node, input_node};
  auto reshape = func_graph->NewCNode(op_inputs);
  reshape->set_fullname_with_scope(input_node->fullname_with_scope() + "_transpose");
  return reshape;
}

bool Process(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto primitive_c = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(cnode->input(0));
    if (primitive_c != nullptr) {
      auto op_type = (schema::PrimitiveType)primitive_c->Type();
      if (op_type != schema::PrimitiveType_Pad || cnode->inputs().size() != lite::kTripleNum) {
        continue;
      }
      // get the second input node whose output is the padding parameter of pad.
      auto input_node = cnode->input(2);
      if (!input_node->isa<CNode>()) {
        continue;
      }
      // reshape the padding of pad operator to 2 x 4.
      std::vector<int> shape_pre = {2, 4};
      auto reshape_pre = NewReshapeOpNode(func_graph, input_node, shape_pre);
      std::vector<int> perm = {1, 0};
      auto transpose = NewTransposeOpNode(func_graph, reshape_pre, perm);
      // reshape the padding of pad operator to -1.
      std::vector<int> shape_pos = {-1};
      auto reshape_pos = NewReshapeOpNode(func_graph, transpose, shape_pos);
      manager->Replace(input_node, reshape_pos);
    } else {
      auto fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
      if (fg != nullptr) {
        if (!Process(fg)) {
          MS_LOG(ERROR) << "Process OnnxPadAdjustPass failed.";
          return false;
        }
      } else {
        MS_LOG(ERROR) << "primitive_c is nullptr: " << cnode->fullname_with_scope();
        return false;
      }
    }
  }
  return true;
}

bool OnnxPadAdjustPass::Run(const FuncGraphPtr &func_graph) {
  if (!Process(func_graph)) {
    MS_LOG(ERROR) << "Run OnnxPadAdjustPass failed.";
    return false;
  }
  return true;
}
}  // namespace mindspore::opt
