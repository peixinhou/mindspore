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
#include "tools/converter/legacy_optimizer/fusion/transpose_reshape_fusion_pass.h"
#include <algorithm>
#include <map>
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "tools/common/graph_util.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
CNodeT *TransposeReshapeFusionPass::GetOneRefNode(schema::MetaGraphT *graph, CNodeT *node, size_t ref_node_input_num) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(node != nullptr);
  if (node->outputIndex.size() != 1) {
    return nullptr;
  }
  auto output_tensor = node->outputIndex[0];
  CNodeT *ret = nullptr;
  int ref_count = 0;
  for (size_t i = 0; i < graph->nodes.size(); ++i) {
    auto ref_node = graph->nodes[i].get();
    if (ref_node->inputIndex.size() == ref_node_input_num && ref_node->inputIndex[0] == output_tensor) {
      ret = ref_node;
      ref_count++;
    }
  }
  if (ref_count == 1) {
    return ret;
  }
  return nullptr;
}

bool TransposeReshapeFusionPass::IsTransposeReshapeParams(const int32_t *shape1_data,
                                                          const std::vector<int32_t> shape1_dims,
                                                          const int32_t *shape2_data,
                                                          const std::vector<int32_t> shape2_dims) {
  MS_ASSERT(shape1_data != nullptr);
  MS_ASSERT(shape2_data != nullptr);
  if (!(shape1_dims.size() == 1 && shape1_dims[0] == 5 && shape2_dims.size() == 1 && shape2_dims[0] == 4)) {
    MS_LOG(DEBUG) << "not special case1 pattern";
    return false;
  }
  if (shape1_data[0] != shape2_data[0] || (shape1_data[1] * shape1_data[2] != shape2_data[1] && shape2_data[1] != -1) ||
      shape1_data[3] != shape2_data[2] || shape1_data[4] != shape2_data[3]) {
    MS_LOG(DEBUG) << "not special case1 pattern";
    return false;
  }
  return true;
}

STATUS TransposeReshapeFusionPass::FuseNodes(schema::MetaGraphT *graph, CNodeT *transpose1, CNodeT *reshape1,
                                             CNodeT *transpose2, CNodeT *reshape2, CNodeT *transpose3) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(transpose1 != nullptr);
  MS_ASSERT(reshape1 != nullptr);
  MS_ASSERT(transpose2 != nullptr);
  MS_ASSERT(reshape2 != nullptr);
  MS_ASSERT(transpose3 != nullptr);
  auto transpose1_prim = transpose1->primitive->value.AsTranspose();
  auto reshape1_prim = reshape1->primitive->value.AsReshape();
  auto transpose2_prim = transpose2->primitive->value.AsTranspose();
  auto reshape2_prim = reshape2->primitive->value.AsReshape();
  auto transpose3_prim = transpose3->primitive->value.AsTranspose();
  MS_ASSERT(transpose1_prim != nullptr);
  MS_ASSERT(reshape1_prim != nullptr);
  MS_ASSERT(transpose2_prim != nullptr);
  MS_ASSERT(reshape2_prim != nullptr);
  MS_ASSERT(transpose3_prim != nullptr);
  if (transpose1_prim->perm != std::vector<int32_t>{0, 3, 1, 2} ||
      transpose2_prim->perm != std::vector<int32_t>{0, 2, 1, 3, 4} ||
      transpose3_prim->perm != std::vector<int32_t>{0, 2, 3, 1}) {
    MS_LOG(DEBUG) << "not special case1 pattern";
    return RET_OK;
  }
  if (reshape1->inputIndex.size() != 2 || reshape1->inputIndex[1] >= graph->allTensors.size() ||
      reshape2->inputIndex.size() != 2 || reshape2->inputIndex[1] >= graph->allTensors.size()) {
    MS_LOG(DEBUG) << "not special case1 pattern";
    return RET_OK;
  }
  auto shape1_tensor = graph->allTensors[reshape1->inputIndex[1]].get();
  auto shape2_tensor = graph->allTensors[reshape2->inputIndex[1]].get();
  if (shape1_tensor->nodeType != NodeType_ValueNode || shape2_tensor->nodeType != NodeType_ValueNode) {
    MS_LOG(DEBUG) << "not special case1 pattern";
    return RET_OK;
  }
  auto shape1_data = reinterpret_cast<int32_t *>(shape1_tensor->data.data());
  auto shape1_dims = graph->allTensors[reshape1->inputIndex[1]]->dims;
  auto shape2_data = reinterpret_cast<int32_t *>(shape2_tensor->data.data());
  auto shape2_dims = graph->allTensors[reshape2->inputIndex[1]]->dims;
  if (!IsTransposeReshapeParams(shape1_data, shape1_dims, shape2_data, shape2_dims)) {
    MS_LOG(DEBUG) << "not special case1 pattern";
    return RET_OK;
  }
  std::vector<int32_t> new_shape1{shape1_data[0], shape1_data[3], shape1_data[4], shape1_data[1], shape1_data[2]};
  std::copy(new_shape1.data(), new_shape1.data() + 5, shape1_data);
  std::vector<int64_t> new_shape2{shape2_data[0], shape2_data[2], shape2_data[3], shape2_data[1]};
  std::copy(new_shape2.data(), new_shape2.data() + 4, shape2_data);
  transpose2_prim->perm = std::vector<int32_t>{0, 1, 2, 4, 3};
  reshape1_prim->format = schema::Format_NHWC;
  reshape2_prim->format = schema::Format_NHWC;
  auto tensor1 = graph->allTensors[transpose2->inputIndex[0]].get();
  auto tensor2 = graph->allTensors[reshape2->inputIndex[0]].get();
  auto tensor3 = graph->allTensors[transpose3->inputIndex[0]].get();
  auto tensor4 = graph->allTensors[transpose3->outputIndex[0]].get();
  MS_ASSERT(tensor1 != nullptr);
  MS_ASSERT(tensor2 != nullptr);
  MS_ASSERT(tensor3 != nullptr);
  MS_ASSERT(tensor4 != nullptr);
  tensor1->dims = new_shape1;
  tensor2->dims = std::vector<int32_t>{shape1_data[0], shape1_data[1], shape1_data[2], shape1_data[4], shape1_data[3]};
  tensor3->dims = tensor4->dims;
  auto status = IsolateOneWayNode(graph, transpose1);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "IsolateOneWayNode failed, node: " << transpose1->name << ", error: " << status;
    return status;
  }
  status = IsolateOneWayNode(graph, transpose3);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "IsolateOneWayNode failed, node: " << transpose3->name << ", error: " << status;
    return status;
  }
  MS_LOG(INFO) << "3 transpose 2 reshape fusion success";
  return RET_OK;
}

STATUS TransposeReshapeFusionPass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  for (size_t i = 0; i < graph->nodes.size(); ++i) {
    auto transpose1 = graph->nodes[i].get();
    if (transpose1->primitive->value.type != schema::PrimitiveType_Transpose) {
      continue;
    }
    auto reshape1 = GetOneRefNode(graph, transpose1, 2);
    if (reshape1 == nullptr || reshape1->primitive->value.type != schema::PrimitiveType_Reshape) {
      continue;
    }
    auto transpose2 = GetOneRefNode(graph, reshape1, 1);
    if (transpose2 == nullptr || transpose2->primitive->value.type != schema::PrimitiveType_Transpose) {
      continue;
    }
    auto reshape2 = GetOneRefNode(graph, transpose2, 2);
    if (reshape2 == nullptr || reshape2->primitive->value.type != schema::PrimitiveType_Reshape) {
      continue;
    }
    auto transpose3 = GetOneRefNode(graph, reshape2, 1);
    if (transpose3 == nullptr || transpose3->primitive->value.type != schema::PrimitiveType_Transpose) {
      continue;
    }
    auto status = FuseNodes(graph, transpose1, reshape1, transpose2, reshape2, transpose3);
    if (status != RET_OK) {
      return status;
    }
  }
  return RET_OK;
}
}  // namespace lite

}  // namespace mindspore
