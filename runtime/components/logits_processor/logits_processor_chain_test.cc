// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/components/logits_processor/logits_processor_chain.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/logits_processor/constrained_decoding/constrained_decoder.h"
#include "runtime/components/logits_processor/constrained_decoding/fake_constraint.h"
#include "runtime/components/logits_processor/logits_processor.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::Return;

class MockLogitsProcessor : public LogitsProcessor {
 public:
  MOCK_METHOD(absl::Status, UpdateState,
              (const ::litert::TensorBuffer& next_token_ids), (override));
  MOCK_METHOD(absl::Status, UpdateState, (absl::Span<int> next_token_ids),
              (override));
  MOCK_METHOD(absl::Status, ProcessLogits, (::litert::TensorBuffer & logits),
              (override));
  MOCK_METHOD(absl::Status, ProcessLogits,
              (absl::Span<float> logits,
               absl::Span<const ::litert::Layout::Dim> logits_dims),
              (override));
  MOCK_METHOD(absl::Status, ProcessLogits,
              (absl::Span<tflite::half> logits,
               absl::Span<const ::litert::Layout::Dim> logits_dims),
              (override));
};

TEST(LogitsProcessorChainTest, EmptyChainWorks) {
  LogitsProcessorChain chain;
  EXPECT_TRUE(chain.empty());
  EXPECT_EQ(chain.size(), 0);
  EXPECT_EQ(chain.GetConstraintDecoder(), nullptr);

  std::vector<int> tokens = {1, 2, 3};
  EXPECT_OK(chain.UpdateState(absl::MakeSpan(tokens)));

  std::vector<float> logits = {0.1f, 0.2f};
  std::vector<::litert::Layout::Dim> dims = {1, 1, 2};
  EXPECT_OK(chain.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));
}

TEST(LogitsProcessorChainTest, HandlesMultipleProcessors) {
  LogitsProcessorChain chain;
  auto proc1 = std::make_unique<MockLogitsProcessor>();
  auto proc2 = std::make_unique<MockLogitsProcessor>();

  std::vector<int> tokens = {1, 2, 3};
  std::vector<float> logits = {0.1f, 0.2f};
  std::vector<::litert::Layout::Dim> dims = {1, 1, 2};

  // We are going to simulate proc1 modifying the logits, so proc2 should see
  // the adjusted logits.
  std::vector<float> adjusted_logits = {0.5f, 0.6f};

  // Enforce sequential calls for UpdateState.
  {
    ::testing::InSequence s;
    EXPECT_CALL(*proc1, UpdateState(absl::Span<int>(tokens)))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*proc2, UpdateState(absl::Span<int>(tokens)))
        .WillOnce(Return(absl::OkStatus()));
  }

  // Enforce sequential calls for ProcessLogits.
  {
    ::testing::InSequence s;
    EXPECT_CALL(*proc1,
                ProcessLogits(absl::Span<float>(logits),
                              absl::Span<const ::litert::Layout::Dim>(dims)))
        .WillOnce([&adjusted_logits](absl::Span<float> logits,
                                     absl::Span<const ::litert::Layout::Dim>) {
          logits[0] = adjusted_logits[0];
          logits[1] = adjusted_logits[1];
          return absl::OkStatus();
        });
    EXPECT_CALL(*proc2,
                ProcessLogits(absl::Span<float>(adjusted_logits),
                              absl::Span<const ::litert::Layout::Dim>(dims)))
        .WillOnce(Return(absl::OkStatus()));
  }

  chain.AddProcessor(std::move(proc1));
  chain.AddProcessor(std::move(proc2));

  EXPECT_FALSE(chain.empty());
  EXPECT_EQ(chain.size(), 2);

  EXPECT_OK(chain.UpdateState(absl::MakeSpan(tokens)));
  EXPECT_OK(chain.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));

  // Verify that the original logits vector has been adjusted correctly through
  // the chain.
  EXPECT_EQ(logits[0], 0.5f);
  EXPECT_EQ(logits[1], 0.6f);
}

TEST(LogitsProcessorChainTest, HaltsOnUpdateStateError) {
  LogitsProcessorChain chain;
  auto proc1 = std::make_unique<MockLogitsProcessor>();
  auto proc2 = std::make_unique<MockLogitsProcessor>();

  std::vector<int> tokens = {1, 2, 3};

  EXPECT_CALL(*proc1, UpdateState(absl::Span<int>(tokens)))
      .WillOnce(Return(absl::InternalError("Test error")));
  EXPECT_CALL(*proc2, UpdateState(absl::Span<int>(tokens))).Times(0);

  chain.AddProcessor(std::move(proc1));
  chain.AddProcessor(std::move(proc2));

  EXPECT_FALSE(chain.UpdateState(absl::MakeSpan(tokens)).ok());
}

TEST(LogitsProcessorChainTest, HaltsOnProcessLogitsError) {
  LogitsProcessorChain chain;
  auto proc1 = std::make_unique<MockLogitsProcessor>();
  auto proc2 = std::make_unique<MockLogitsProcessor>();

  std::vector<float> logits = {0.1f, 0.2f};
  std::vector<::litert::Layout::Dim> dims = {1, 1, 2};

  EXPECT_CALL(*proc1,
              ProcessLogits(absl::Span<float>(logits),
                            absl::Span<const ::litert::Layout::Dim>(dims)))
      .WillOnce(Return(absl::InternalError("Test error")));
  EXPECT_CALL(*proc2,
              ProcessLogits(absl::Span<float>(logits),
                            absl::Span<const ::litert::Layout::Dim>(dims)))
      .Times(0);

  chain.AddProcessor(std::move(proc1));
  chain.AddProcessor(std::move(proc2));

  EXPECT_FALSE(
      chain.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)).ok());
}

TEST(LogitsProcessorChainTest, ReturnsLastAddedConstraintDecoder) {
  LogitsProcessorChain chain;
  EXPECT_EQ(chain.GetConstraintDecoder(), nullptr);

  // Add a normal logits processor
  chain.AddProcessor(std::make_unique<MockLogitsProcessor>());
  EXPECT_EQ(chain.GetConstraintDecoder(), nullptr);

  // Add first ConstrainedDecoder
  auto constraint1 = FakeConstraint({1}, 10);
  auto cd1 = std::make_unique<ConstrainedDecoder>(&constraint1, 1);
  auto ptr1 = cd1.get();
  chain.AddProcessor(std::move(cd1));
  EXPECT_EQ(chain.GetConstraintDecoder(), ptr1);

  // Add another ConstrainedDecoder
  auto constraint2 = FakeConstraint({2}, 10);
  auto cd2 = std::make_unique<ConstrainedDecoder>(&constraint2, 1);
  auto ptr2 = cd2.get();
  chain.AddProcessor(std::move(cd2));

  // It should return the last added one based on our code's behavior
  EXPECT_EQ(chain.GetConstraintDecoder(), ptr2);
}

}  // namespace
}  // namespace litert::lm
