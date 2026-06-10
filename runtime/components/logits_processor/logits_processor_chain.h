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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_LOGITS_PROCESSOR_CHAIN_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_LOGITS_PROCESSOR_CHAIN_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/logits_processor/constrained_decoding/constrained_decoder.h"
#include "runtime/components/logits_processor/logits_processor.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
// `LogitsProcessorChain` acts as a composite runtime container for multiple
// `LogitsProcessor`s executing synchronously in serial.
//
// Individual logit modifications—such as strict grammar constraints
// (e.g. ConstrainedDecoder), token banning, or repetition penalizations—
// operate sequentially. The output logit modifications applied from one
// processor propagate to the next.
//
// In addition to implementing the core `LogitsProcessor` interfaces, the
// chain provides explicit getter exposure to internally held processors
// like `ConstrainedDecoder` to accommodate backends requiring legacy direct
// access to mask application resources.
class LogitsProcessorChain : public LogitsProcessor {
 public:
  LogitsProcessorChain() = default;
  ~LogitsProcessorChain() override = default;

  LogitsProcessorChain(const LogitsProcessorChain&) = delete;
  LogitsProcessorChain& operator=(const LogitsProcessorChain&) = delete;

  LogitsProcessorChain(LogitsProcessorChain&&) noexcept = default;
  LogitsProcessorChain& operator=(LogitsProcessorChain&&) noexcept = default;

  absl::Status ProcessLogits(::litert::TensorBuffer& logits) override;

  absl::Status ProcessLogits(
      absl::Span<float> logits,
      absl::Span<const ::litert::Layout::Dim> logits_dims) override;

  absl::Status ProcessLogits(
      absl::Span<tflite::half> logits,
      absl::Span<const ::litert::Layout::Dim> logits_dims) override;

  absl::Status UpdateState(
      const ::litert::TensorBuffer& next_token_ids) override;

  absl::Status UpdateState(absl::Span<int> next_token_ids) override;

  // Adds a logits processor to the chain string.
  void AddProcessor(std::unique_ptr<LogitsProcessor> processor);

  // Adds a constrained decoder to the chain string. This is a special case for
  // `LlmLiteRTExecutor` & `LlmGpuArtisanExecutor` to retrieve the constraint
  // decoder from the chain.
  void AddProcessor(std::unique_ptr<ConstrainedDecoder> processor);

  // Returns true if the chain is empty.
  bool empty() const;

  // Returns the number of logit processors in the chain.
  int size() const;

  // Returns the last added constraint decoder.
  ConstrainedDecoder* GetConstraintDecoder() const;

 private:
  std::vector<std::unique_ptr<LogitsProcessor>> processors_;
  ConstrainedDecoder* constraint_decoder_ = nullptr;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_LOGITS_PROCESSOR_CHAIN_H_
