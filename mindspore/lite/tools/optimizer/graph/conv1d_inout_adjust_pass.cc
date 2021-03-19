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
#include "tools/optimizer/graph/conv1d_inout_adjust_pass.h"
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include "mindspore/lite/include/errorcode.h"
#include "src/ops/primitive_c.h"

namespace mindspore::opt {
ValueNodePtr Conv1DInOutAdjustPass::CreateNewValueNode(void *attr, const schema::PrimitiveType &op_type) {
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

CNodePtr Conv1DInOutAdjustPass::NewUnsqueezeOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr input_node,
                                                   const schema::Conv2DT *conv2d_attr) {
  auto attr = std::make_unique<schema::UnsqueezeT>();
  if (conv2d_attr->format == schema::Format_NWC) {
    attr->axis = {1};
  } else if (conv2d_attr->format == schema::Format_NCW) {
    attr->axis = {2};
  }

  ValueNodePtr value_node = CreateNewValueNode(attr.release(), schema::PrimitiveType_Unsqueeze);
  std::vector<AnfNodePtr> op_inputs = {value_node, input_node};
  auto unsqueeze = func_graph->NewCNode(op_inputs);
  unsqueeze->set_fullname_with_scope(input_node->fullname_with_scope() + "_unsqueeze");
  return unsqueeze;
}

CNodePtr Conv1DInOutAdjustPass::NewSqueezeOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr input_node,
                                                 schema::Conv2DT *conv2d_attr) {
  auto attr = std::make_unique<schema::SqueezeT>();
  if (conv2d_attr->format == schema::Format_NWC) {
    attr->axis = {1};
    conv2d_attr->format = schema::Format_NHWC;
  } else if (conv2d_attr->format == schema::Format_NCW) {
    attr->axis = {2};
    conv2d_attr->format = schema::Format_NCHW;
  }

  ValueNodePtr value_node = CreateNewValueNode(attr.release(), schema::PrimitiveType_Squeeze);
  std::vector<AnfNodePtr> op_inputs = {value_node, input_node};
  auto squeeze = func_graph->NewCNode(op_inputs);
  squeeze->set_fullname_with_scope(input_node->fullname_with_scope() + "_squeeze");
  return squeeze;
}

bool Conv1DInOutAdjustPass::Process(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto primitive_c = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(cnode->input(0));
    if (primitive_c != nullptr) {
      auto op_type = (schema::PrimitiveType)primitive_c->Type();
      if (op_type != schema::PrimitiveType_Conv2D) {
        continue;
      }
      auto conv2d_attr = reinterpret_cast<schema::Conv2DT *>(primitive_c->primitiveT()->value.value);
      if (conv2d_attr->format != schema::Format_NWC && conv2d_attr->format != schema::Format_NCW) {
        continue;
      }

      auto input_node = cnode->input(1);
      auto unsqueeze = NewUnsqueezeOpNode(func_graph, input_node, conv2d_attr);
      manager->Replace(input_node, unsqueeze);
      auto squeeze = NewSqueezeOpNode(func_graph, cnode, conv2d_attr);
      manager->Replace(cnode, squeeze);
    } else {
      auto fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
      if (fg != nullptr) {
        if (!Process(fg)) {
          MS_LOG(ERROR) << "Process Conv1DInOutAdjustPass failed.";
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

bool Conv1DInOutAdjustPass::Run(const FuncGraphPtr &func_graph) {
  if (!Process(func_graph)) {
    MS_LOG(ERROR) << "Run Conv1DInOutAdjustPass failed.";
    return false;
  }
  return true;
}
}  // namespace mindspore::opt
