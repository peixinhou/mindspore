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

#include "src/runtime/allocator.h"
#include <utility>
#include "src/common/log_adapter.h"

namespace mindspore::lite {
std::shared_ptr<Allocator> Allocator::Create() {
  return std::shared_ptr<Allocator>(new (std::nothrow) DefaultAllocator());
}

DefaultAllocator::DefaultAllocator() = default;

DefaultAllocator::~DefaultAllocator() { Clear(); }

void DefaultAllocator::SetContext(const AllocatorContext &ctx) {
  lock_flag_ = ctx.lockFlag;
  shift_factor_ = ctx.shiftFactor;
}

void DefaultAllocator::Lock() {
  if (lock_flag_) {
    lock_.lock();
  }
}

void DefaultAllocator::UnLock() {
  if (lock_flag_) {
    lock_.unlock();
  }
}

void *DefaultAllocator::Malloc(size_t size) {
  if (size > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "MallocData out of max_size, size: " << size;
    return nullptr;
  }
  Lock();
  auto iter = free_list_.lower_bound(size);
  if (iter != free_list_.end() && (iter->second->size >= size) && (iter->second->size < (size << shift_factor_))) {
    auto membuf = iter->second;
    free_list_.erase(iter);
    allocated_list_[membuf->buf] = membuf;
    UnLock();
    return membuf->buf;
  }

  std::unique_ptr<MemBuf> membuf(reinterpret_cast<MemBuf *>(malloc(sizeof(MemBuf) + size)));
  if (membuf == nullptr) {
    MS_LOG(ERROR) << "malloc membuf return nullptr";
    UnLock();
    return nullptr;
  }
  membuf->size = size;
  membuf->buf = reinterpret_cast<char *>(membuf.get()) + sizeof(MemBuf);
  auto bufPtr = membuf->buf;
  allocated_list_[bufPtr] = membuf.release();
  UnLock();
  return bufPtr;
}

void DefaultAllocator::Free(void *buf) {
  if (buf == nullptr) {
    return;
  }
  Lock();
  auto iter = allocated_list_.find(buf);
  if (iter != allocated_list_.end()) {
    auto membuf = iter->second;
    allocated_list_.erase(iter);
    free_list_.insert(std::make_pair(membuf->size, membuf));
    UnLock();
    return;
  }
  UnLock();
  free(buf);
  buf = nullptr;
}

size_t DefaultAllocator::GetTotalSize() {
  Lock();
  size_t totalSize = 0;

  for (auto &it : allocated_list_) {
    auto membuf = it.second;
    totalSize += membuf->size;
  }

  for (auto &it : free_list_) {
    auto membuf = it.second;
    totalSize += membuf->size;
  }
  UnLock();
  return totalSize;
}

void DefaultAllocator::Clear() {
  Lock();

  for (auto &it : allocated_list_) {
    free(it.second);
  }
  allocated_list_.clear();

  for (auto &it : free_list_) {
    free(it.second);
  }
  free_list_.clear();
  UnLock();
}
}  // namespace mindspore::lite
