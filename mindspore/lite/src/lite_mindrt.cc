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

#include <utility>
#include "src/lite_mindrt.h"
#include "mindrt/include/mindrt.hpp"

namespace mindspore::lite {
int LiteOpActor::CompileArrow() {
  int outTensorSize = static_cast<int>(kernel_->out_tensors().size());
  for (int i = 0; i < outTensorSize; i++) {
    for (auto out : kernel_->out_kernels()) {
      int inTensorSize = static_cast<int>(out->in_tensors().size());
      int to_input_index = -1;
      for (int j = 0; j < inTensorSize; j++) {
        if (kernel_->out_tensors()[i] == out->in_tensors()[j]) {
          to_input_index = j;
          break;
        }
      }
      if (to_input_index == -1) {
        continue;
      }
      auto id = out->name() + this->GetAID().Url();
      auto arrow = std::make_shared<OpArrow>(i, AID(id), to_input_index);
      if (arrow == nullptr) {
        MS_LOG(ERROR) << "create OpArrow failed, out kernel: " << out->name();
        return RET_ERROR;
      }
      output_op_arrows_.emplace_back(std::move(arrow));
    }
  }
  return RET_OK;
}

void LiteOpActor::AsyncOutput(OpContext<Tensor> *context) {
  for (auto op_arrow : output_op_arrows_) {
    auto data = context->output_data_->at(op_arrow->from_output_index_);
    Async(op_arrow->to_op_id_, &mindspore::OpActor<Tensor>::RunOpData, data, context);
  }
  return;
}

void LiteOpActor::AddResultIndex(size_t index) {
  results_index_.push_back(index);
  return;
}

void LiteOpActor::SetOutputData(OpContext<Tensor> *context) {
  for (auto index : results_index_) {
    context->SetResult(index, RET_OK);
  }
}

int MindrtInit() { return mindspore::Initialize("tcp://127.0.0.1:8080", "", "", "", 1); }

void MindrtTerminate(std::vector<std::shared_ptr<LiteOpActor>> actor_list) {
  for (auto actor : actor_list) {
    mindspore::Terminate(actor->GetAID());
  }
  return;
}

std::vector<std::shared_ptr<LiteOpActor>> CreateOpActor(const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<std::shared_ptr<LiteOpActor>> actors;
  for (auto kernel : kernels) {
    auto actor = std::make_shared<LiteOpActor>(kernel);
    if (actor == nullptr) {
      MS_LOG(ERROR) << "create LiteOpActor failed: " << kernel->name();
      actors.clear();
      return actors;
    }
    auto aid = mindspore::Spawn(actor);
    actors.push_back(actor);
  }

  return actors;
}

}  // namespace mindspore::lite
