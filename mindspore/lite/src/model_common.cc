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
#include "src/model_common.h"
#include "include/version.h"
#include "src/ops/while.h"
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore::lite {
bool ConvertNodes(const schema::MetaGraph *meta_graph, Model *model) {
  if (model == nullptr || meta_graph == nullptr || meta_graph->nodes() == nullptr) {
    MS_LOG(ERROR) << "model or meta_graph is invalid, please check your model file.";
    return false;
  }
  for (size_t i = 0; i < meta_graph->nodes()->size(); ++i) {
    auto *node = new (std::nothrow) Model::Node();
    if (node == nullptr) {
      MS_LOG(ERROR) << "new node fail!";
      return false;
    }
    auto c_node = meta_graph->nodes()->GetAs<schema::CNode>(i);
    if (c_node == nullptr || c_node->primitive() == nullptr || c_node->name() == nullptr ||
        c_node->inputIndex() == nullptr) {
      MS_LOG(ERROR) << "c_node is invalid.";
      delete node;
      return false;
    }
    auto src_prim = c_node->primitive();
#ifdef PRIMITIVE_WRITEABLE
    node->primitive_ = PrimitiveC::Create(const_cast<schema::Primitive *>(src_prim));
#else
    auto primitive = const_cast<schema::Primitive *>(src_prim);
    node->primitive_ = OpsRegistry::GetInstance()->getPrimitiveCreator(primitive->value_type())(primitive);
#endif
    if (node->primitive_ == nullptr) {
      MS_LOG(ERROR) << "unpack primitive == nullptr!";
      delete node;
      return false;
    }
    node->primitive_->set_quant_type(c_node->quantType());
    node->name_ = c_node->name()->c_str();
    node->node_type_ = c_node->nodeType();
    auto count = c_node->inputIndex()->size();
    for (uint32_t j = 0; j < count; ++j) {
      node->input_indices_.push_back(c_node->inputIndex()->Get(j));
    }
    if (c_node->outputIndex() != nullptr) {
      count = c_node->outputIndex()->size();
      for (uint32_t j = 0; j < count; ++j) {
        node->output_indices_.push_back(c_node->outputIndex()->Get(j));
      }
    }
    model->all_nodes_.push_back(node);
  }
  return true;
}

bool ConvertTensors(const schema::MetaGraph *meta_graph, Model *model) {
  if (model == nullptr || meta_graph == nullptr || meta_graph->allTensors() == nullptr) {
    MS_LOG(ERROR) << "model or meta_graph is invalid, please check your model file.";
    return false;
  }
  auto tensor_count = meta_graph->allTensors()->size();
  for (uint32_t i = 0; i < tensor_count; ++i) {
    auto *tensor = meta_graph->allTensors()->GetAs<schema::Tensor>(i);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << i << "the tensor in model is nullptr";
      return false;
    }
    model->all_tensors_.push_back(const_cast<mindspore::schema::Tensor *>(tensor));
  }
  return true;
}

int ConvertSubGraph(const schema::SubGraph *sub_graph, Model *model) {
  if (model == nullptr || sub_graph == nullptr) {
    MS_LOG(ERROR) << "model or sub_graph is null.";
    return RET_ERROR;
  }
  if (sub_graph->name() == nullptr || sub_graph->inputIndices() == nullptr || sub_graph->outputIndices() == nullptr ||
      sub_graph->nodeIndices() == nullptr || sub_graph->tensorIndices() == nullptr) {
    MS_LOG(ERROR) << "sub_graph is invalid.";
    return RET_ERROR;
  }
  auto *sub_graph_temp = new (std::nothrow) Model::SubGraph();
  if (sub_graph_temp == nullptr) {
    MS_LOG(ERROR) << "new subGraph fail!";
    return RET_ERROR;
  }
  sub_graph_temp->name_ = sub_graph->name()->c_str();
  auto in_count = sub_graph->inputIndices()->size();
  for (uint32_t i = 0; i < in_count; ++i) {
    sub_graph_temp->input_indices_.push_back(sub_graph->inputIndices()->Get(i));
  }
  auto out_count = sub_graph->outputIndices()->size();
  for (uint32_t i = 0; i < out_count; ++i) {
    sub_graph_temp->output_indices_.push_back(sub_graph->outputIndices()->Get(i));
  }
  auto node_count = sub_graph->nodeIndices()->size();
  for (uint32_t i = 0; i < node_count; ++i) {
    sub_graph_temp->node_indices_.push_back(sub_graph->nodeIndices()->Get(i));
  }
  auto tensor_count = sub_graph->tensorIndices()->size();
  for (uint32_t i = 0; i < tensor_count; ++i) {
    sub_graph_temp->tensor_indices_.push_back(sub_graph->tensorIndices()->Get(i));
  }
  model->sub_graphs_.push_back(sub_graph_temp);
  return RET_OK;
}

int MetaGraphMappingSubGraph(const mindspore::schema::MetaGraph *meta_graph, Model *model) {
  if (model == nullptr || meta_graph == nullptr) {
    MS_LOG(ERROR) << "model or meta_graph is null.";
    return RET_ERROR;
  }
  if (meta_graph->inputIndex() == nullptr || meta_graph->outputIndex() == nullptr || meta_graph->nodes() == nullptr ||
      meta_graph->allTensors() == nullptr) {
    MS_LOG(ERROR) << "meta_graph is invalid, please check your model file.";
    return RET_ERROR;
  }
  auto *sub_graph_temp = new (std::nothrow) Model::SubGraph();
  if (sub_graph_temp == nullptr) {
    MS_LOG(ERROR) << "new subGraph fail!";
    return RET_ERROR;
  }
  if (meta_graph->name() != nullptr) {
    sub_graph_temp->name_ = meta_graph->name()->c_str();
  }
  auto in_count = meta_graph->inputIndex()->size();
  for (uint32_t i = 0; i < in_count; ++i) {
    sub_graph_temp->input_indices_.push_back(meta_graph->inputIndex()->Get(i));
  }
  auto out_count = meta_graph->outputIndex()->size();
  for (uint32_t i = 0; i < out_count; ++i) {
    sub_graph_temp->output_indices_.push_back(meta_graph->outputIndex()->Get(i));
  }
  auto node_count = meta_graph->nodes()->size();
  for (uint32_t i = 0; i < node_count; ++i) {
    sub_graph_temp->node_indices_.push_back(i);
  }
  auto tensor_count = meta_graph->allTensors()->size();
  for (uint32_t i = 0; i < tensor_count; ++i) {
    sub_graph_temp->tensor_indices_.push_back(i);
  }
  model->sub_graphs_.push_back(sub_graph_temp);
  return RET_OK;
}

int NodeVerify(const Model &model) {
  auto tensor_size = model.all_tensors_.size();
  uint32_t subGraph_size = model.sub_graphs_.size();

  for (auto &node : model.all_nodes_) {
    if (node == nullptr || node->primitive_ == nullptr) {
      MS_LOG(ERROR) << "node or its primitive_ is null.";
      return RET_ERROR;
    }
    if (std::any_of(node->input_indices_.begin(), node->input_indices_.end(),
                    [&tensor_size](const uint32_t &idx) { return idx >= tensor_size; })) {
      MS_LOG(ERROR) << "Index of node->input_indices_ is beyond size.";
      return RET_ERROR;
    }
    if (std::any_of(node->output_indices_.begin(), node->output_indices_.end(),
                    [&tensor_size](const uint32_t &idx) { return idx >= tensor_size; })) {
      MS_LOG(ERROR) << "Index of node->output_indices_ is beyond size.";
      return RET_ERROR;
    }

    auto prim = node->primitive_;
    if (prim->Type() == schema::PrimitiveType_While) {
      auto whileOp = reinterpret_cast<mindspore::lite::While *>(const_cast<mindspore::lite::PrimitiveC *>(prim));
      if (whileOp == nullptr) {
        MS_LOG(ERROR) << "whileOp is null.";
        return RET_ERROR;
      }
      if (static_cast<uint32_t>(whileOp->GetBodySubgraphIndex()) >= subGraph_size ||
          static_cast<uint32_t>(whileOp->GetCondSubgraphIndex()) >= subGraph_size) {
        MS_LOG(ERROR) << "index of subGraph is beyond subGraph_size.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int SubGraphVerify(const Model &model) {
  auto tensor_size = model.all_tensors_.size();
  auto node_size = model.all_nodes_.size();

  if (!model.sub_graphs_.empty()) {
    for (auto &graph : model.sub_graphs_) {
      if (graph == nullptr) {
        MS_LOG(ERROR) << "graph is null.";
        return RET_ERROR;
      }
      if (std::any_of(graph->input_indices_.begin(), graph->input_indices_.end(),
                      [&tensor_size](const uint32_t &idx) { return idx >= tensor_size; })) {
        MS_LOG(ERROR) << "Index of graph->input_indices_ is beyond tensor_size.";
        return RET_ERROR;
      }
      if (std::any_of(graph->output_indices_.begin(), graph->output_indices_.end(),
                      [&tensor_size](const uint32_t &idx) { return idx >= tensor_size; })) {
        MS_LOG(ERROR) << "Index of graph->output_indices_ is beyond tensor_size.";
        return RET_ERROR;
      }
      if (std::any_of(graph->tensor_indices_.begin(), graph->tensor_indices_.end(),
                      [&tensor_size](const uint32_t &idx) { return idx >= tensor_size; })) {
        MS_LOG(ERROR) << "Index of graph->tensor_indices_ is beyond tensor_size.";
        return RET_ERROR;
      }
      if (std::any_of(graph->node_indices_.begin(), graph->node_indices_.end(),
                      [&node_size](const uint32_t &idx) { return idx >= node_size; })) {
        MS_LOG(ERROR) << "Index of graph->node_indices_ is beyond node_size.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

bool ModelVerify(const Model &model) { return NodeVerify(model) == RET_OK && SubGraphVerify(model) == RET_OK; }

Model *ImportFromBuffer(const char *model_buf, size_t size, bool take_buf) {
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "The model buf is nullptr";
    return nullptr;
  }
  flatbuffers::Verifier verify((const uint8_t *)model_buf, size);
  if (!schema::VerifyMetaGraphBuffer(verify)) {
    MS_LOG(ERROR) << "The buffer is invalid and fail to create graph.";
    return nullptr;
  }
  auto *model = new (std::nothrow) Model();
  if (model == nullptr) {
    MS_LOG(ERROR) << "new model fail!";
    return nullptr;
  }
  if (take_buf) {
    model->buf = const_cast<char *>(model_buf);
  } else {
    if (size == 0) {
      MS_LOG(ERROR) << "malloc size is equal to 0";
      delete model;
      return nullptr;
    }
    model->buf = reinterpret_cast<char *>(malloc(size));
    if (model->buf == nullptr) {
      MS_LOG(ERROR) << "new inner model buf fail!";
      delete model;
      return nullptr;
    }
    memcpy(model->buf, model_buf, size);
  }

  auto meta_graph = schema::GetMetaGraph(model->buf);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "meta_graph is nullptr!";
    delete model;
    return nullptr;
  }

  if (meta_graph->name() != nullptr) {
    model->name_ = meta_graph->name()->c_str();
  }
  if (meta_graph->version() != nullptr) {
    model->version_ = meta_graph->version()->c_str();
  }

  if (model->version_ != Version()) {
    MS_LOG(WARNING) << "model version is " << model->version_ << ", inference version is " << Version() << " not equal";
  }

  if (!ConvertNodes(meta_graph, model)) {
    delete model;
    return nullptr;
  }

  if (!ConvertTensors(meta_graph, model)) {
    delete model;
    return nullptr;
  }

  if (meta_graph->subGraph() == nullptr) {
    int ret = MetaGraphMappingSubGraph(meta_graph, model);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "converter old version model wrong.";
      delete model;
      return nullptr;
    }
  } else {
    auto sub_graphs = meta_graph->subGraph();
    auto sub_graph_size = sub_graphs->size();
    for (size_t i = 0; i < sub_graph_size; i++) {
      auto sub_graph = sub_graphs->GetAs<schema::SubGraph>(i);
      int ret = ConvertSubGraph(sub_graph, model);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "converter subgraph wrong.";
        delete model;
        return nullptr;
      }
    }
  }
  if (model->sub_graphs_.empty()) {
    delete model;
    return nullptr;
  }
  return ModelVerify(*model) ? model : nullptr;
}
}  // namespace mindspore::lite
