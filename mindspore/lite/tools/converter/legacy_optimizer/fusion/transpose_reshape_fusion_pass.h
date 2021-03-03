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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_FUSION_TRANSPOSE_RESHAPE_FUSION_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_FUSION_TRANSPOSE_RESHAPE_FUSION_H_

#include <vector>
#include <utility>
#include <set>
#include <memory>
#include <map>
#include "tools/converter/optimizer.h"

namespace mindspore {
namespace lite {
class TransposeReshapeFusionPass : public GraphPass {
 public:
  TransposeReshapeFusionPass() = default;

  ~TransposeReshapeFusionPass() override = default;

  STATUS Run(schema::MetaGraphT *graph) override;

 private:
  STATUS FuseNodes(schema::MetaGraphT *graph, CNodeT *transpose1, CNodeT *reshape1, CNodeT *transpose2,
                   CNodeT *reshape2, CNodeT *transpose3);
  bool IsTransposeReshapeParams(const int32_t *shape1_data, const std::vector<int32_t> shape1_dims,
                                const int32_t *shape2_data, const std::vector<int32_t> shape2_dims);
  CNodeT *GetOneRefNode(schema::MetaGraphT *graph, CNodeT *node, size_t ref_node_input_num);
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_FUSION_TRANSPOSE_RESHAPE_FUSION_H_
