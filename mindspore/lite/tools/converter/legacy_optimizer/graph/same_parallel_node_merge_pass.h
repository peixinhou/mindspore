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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_GRAPH_SAME_PARALLEL_NODE_MERGE_PASS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_GRAPH_SAME_PARALLEL_NODE_MERGE_PASS_H_

#include <vector>
#include <utility>
#include <set>
#include <memory>
#include <map>
#include "tools/converter/optimizer.h"

namespace mindspore {
namespace lite {
class SameParallelNodeMergePass : public GraphPass {
 public:
  SameParallelNodeMergePass() = default;

  ~SameParallelNodeMergePass() override = default;

  bool IsSamePrimitive(CNodeT *a, CNodeT *b);

  STATUS MergeNode(schema::MetaGraphT *graph, const std::vector<int> &sub_graph_ids, const int sub_graph_id,
                   const CNodeT *a, CNodeT *b);

  STATUS Merge(const std::vector<int> &sub_graph_ids, schema::MetaGraphT *graph);

  STATUS UpdateIndexes(const std::map<uint32_t, uint32_t> &node_index_map,
                       const std::map<uint32_t, uint32_t> &tensor_index_map, schema::SubGraphT *sub_graph);

  STATUS Run(schema::MetaGraphT *graph) override;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_GRAPH_SAME_PARALLEL_NODE_MERGE_PASS_H_
