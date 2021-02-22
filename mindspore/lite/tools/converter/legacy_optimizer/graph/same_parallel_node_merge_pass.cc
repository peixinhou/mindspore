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
#include "tools/converter/legacy_optimizer/graph/same_parallel_node_merge_pass.h"
#include <algorithm>
#include <map>
#include "tools/common/graph_util.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
namespace {
static const std::vector<schema::PrimitiveType> kNotMergeNodeTypes{
  schema::PrimitiveType_Merge, schema::PrimitiveType_Switch, schema::PrimitiveType_Partial};
}
STATUS SameParallelNodeMergePass::MergeNode(schema::MetaGraphT *graph, const std::vector<int> &sub_graph_ids,
                                            const int sub_graph_id, const CNodeT *a, CNodeT *b) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(a != nullptr);
  MS_ASSERT(b != nullptr);
  MS_ASSERT(a->outputIndex.size() == b->outputIndex.size());
  for (size_t i = 0; i < b->outputIndex.size(); ++i) {
    auto ref_node_ids = GetLinkedPostIdx(*graph, b->outputIndex[i]);
    for (auto ref_node_id : ref_node_ids) {
      if (sub_graph_ids[ref_node_id] != sub_graph_id) {
        continue;
      }
      auto ref_node = graph->nodes[ref_node_id].get();
      for (size_t j = 0; j < ref_node->inputIndex.size(); ++j) {
        if (ref_node->inputIndex[j] == b->outputIndex[i]) {
          ref_node->inputIndex[j] = a->outputIndex[i];
        }
      }
    }
  }

  for (size_t i = 0; i < b->outputIndex.size(); ++i) {
    for (size_t j = 0; j < graph->subGraph[sub_graph_id]->outputIndices.size(); ++j) {
      if (b->outputIndex[i] == graph->subGraph[sub_graph_id]->outputIndices[j]) {
        graph->subGraph[sub_graph_id]->outputIndices[j] = a->outputIndex[i];
      }
    }
  }

  for (size_t i = 0; i < b->outputIndex.size(); ++i) {
    for (size_t j = 0; j < graph->outputIndex.size(); ++j) {
      if (b->outputIndex[i] == graph->outputIndex[j]) {
        graph->outputIndex[j] = a->outputIndex[i];
      }
    }
  }

  auto status = RemoveTensor(graph, b->outputIndex);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "remove tensor failed";
    return status;
  }
  b->inputIndex.clear();
  b->outputIndex.clear();
  return RET_OK;
}

bool SameParallelNodeMergePass::IsSamePrimitive(CNodeT *a, CNodeT *b) {
  MS_ASSERT(a != nullptr);
  MS_ASSERT(b != nullptr);
  if (a->primitive->value.type == b->primitive->value.type) {
    flatbuffers::FlatBufferBuilder fbb1;
    flatbuffers::FlatBufferBuilder fbb2;
    fbb1.Finish(a->primitive->value.Pack(fbb1));
    fbb2.Finish(b->primitive->value.Pack(fbb2));
    if (fbb1.GetSize() == fbb2.GetSize()) {
      if (memcmp(fbb1.GetBufferPointer(), fbb2.GetBufferPointer(), fbb1.GetSize()) == 0) {
        return true;
      }
    }
  }
  return false;
}

STATUS SameParallelNodeMergePass::Merge(const std::vector<int> &sub_graph_ids, schema::MetaGraphT *graph) {
  for (size_t i = 0; i < graph->allTensors.size(); ++i) {
    auto output_node = GetLinkedPostIdx(*graph, i);
    for (size_t j = 0; j < output_node.size(); ++j) {
      auto nodej = graph->nodes[output_node[j]].get();
      MS_ASSERT(nodej != nullptr);
      if (IsContain(kNotMergeNodeTypes, nodej->primitive->value.type)) {
        continue;
      }
      for (size_t t = j + 1; t < output_node.size(); ++t) {
        auto nodet = graph->nodes[output_node[t]].get();
        MS_ASSERT(nodet != nullptr);
        // two nodes must belong to same subgraph
        if (nodet == nodej || nodet->inputIndex != nodej->inputIndex ||
            sub_graph_ids[output_node[j]] != sub_graph_ids[output_node[t]]) {
          continue;
        }
        if (nodej->outputIndex.size() == nodet->outputIndex.size() && IsSamePrimitive(nodej, nodet)) {
          auto status = MergeNode(graph, sub_graph_ids, sub_graph_ids[output_node[j]], nodej, nodet);
          if (status != RET_OK) {
            return status;
          }
          return RET_OK;
        }
      }
    }
  }
  return RET_NO_CHANGE;
}

STATUS SameParallelNodeMergePass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  std::vector<int> sub_graph_ids(graph->nodes.size(), -1);
  for (size_t i = 0; i < graph->subGraph.size(); ++i) {
    auto &sub_graph = graph->subGraph[i];
    for (auto &node_index : sub_graph->nodeIndices) {
      if (node_index >= graph->nodes.size() || sub_graph_ids[node_index] != -1) {
        MS_LOG(ERROR) << "node index error or one node in multi subgraphs";
        return RET_ERROR;
      }
      sub_graph_ids[node_index] = i;
    }
  }
  while (true) {
    auto status = Merge(sub_graph_ids, graph);
    if (status == RET_NO_CHANGE) {
      break;
    }
    if (status != RET_OK) {
      return status;
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
