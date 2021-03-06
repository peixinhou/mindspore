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

#include <memory>
#include <utility>
#include <set>
#include <string>
#include <unordered_map>

#include "ir/anf.h"
#include "tools/optimizer/parallel/dynamic_creator.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/ccsrc/backend/optimizer/common/node_pass.h"

#ifndef MINDSPORE_LITE_SRC_PASS_PARALLEL_PARALLEL_PASS_H_
#define MINDSPORE_LITE_SRC_PASS_PARALLEL_PARALLEL_PASS_H_

namespace mindspore {
namespace opt {
class ParallelPass : public opt::NodePass {
 public:
  explicit ParallelPass(const std::unordered_map<std::string, SplitStrategy> strategys, const int32_t FmkType)
      : NodePass("parallel_pass"), split_strategys_(strategys), FmkType_(FmkType) {}
  ~ParallelPass() override = default;
  AnfNodePtr Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) override;

 private:
  const std::set<PrimitivePtr> PARALLEL_LIST = {prim::kPrimConv2DFusion};
  const std::unordered_map<std::string, std::string> type_string = {{prim::kPrimConv2DFusion->name(), "Conv2D"}};

  bool IsParallelCareNode(const AnfNodePtr &node);
  std::string PrimToString(const PrimitivePtr &prim);

  std::string type_name_;
  std::unordered_map<std::string, SplitStrategy> split_strategys_;
  int32_t FmkType_;
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_PASS_PARALLEL_PARALLEL_PASS_H_
