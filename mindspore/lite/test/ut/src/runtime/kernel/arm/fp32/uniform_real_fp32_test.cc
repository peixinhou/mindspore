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

#include <iostream>
#include "common/common_test.h"
#include "mindspore/lite/nnacl/random_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class TestUniformRealFp32 : public mindspore::CommonTest {
 public:
  TestUniformRealFp32() {}
};

TEST_F(TestUniformRealFp32, UniformReal) {
  lite::Tensor out_tensor0(kNumberTypeFloat32, {10});
  float output_data0[10] = {0};
  out_tensor0.set_data(output_data0);
  std::vector<lite::Tensor *> inputs = {};
  std::vector<lite::Tensor *> outputs = {&out_tensor0};

  RandomParam parameter;
  parameter.op_parameter_.type_ = schema::PrimitiveType_UniformReal;
  parameter.seed_ = 42;
  parameter.seed2_ = 959;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_UniformReal};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc, nullptr);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);
  for (int i = 0; i < 10; ++i) {
    std::cout << output_data0[i] << " ";
  }
  out_tensor0.set_data(nullptr);
}
}  // namespace mindspore
