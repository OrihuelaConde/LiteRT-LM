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

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/logits_processor/constrained_decoding/constrained_decoder.h"
#include "runtime/components/logits_processor/logits_processor.h"
#include "runtime/util/status_macros.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

absl::Status LogitsProcessorChain::ProcessLogits(
    ::litert::TensorBuffer& logits) {
  for (auto& processor : processors_) {
    LITERT_RETURN_IF_ERROR(processor->ProcessLogits(logits));
  }
  return absl::OkStatus();
}

absl::Status LogitsProcessorChain::ProcessLogits(
    absl::Span<float> logits,
    absl::Span<const ::litert::Layout::Dim> logits_dims) {
  for (auto& processor : processors_) {
    LITERT_RETURN_IF_ERROR(processor->ProcessLogits(logits, logits_dims));
  }
  return absl::OkStatus();
}

absl::Status LogitsProcessorChain::ProcessLogits(
    absl::Span<tflite::half> logits,
    absl::Span<const ::litert::Layout::Dim> logits_dims) {
  for (auto& processor : processors_) {
    LITERT_RETURN_IF_ERROR(processor->ProcessLogits(logits, logits_dims));
  }
  return absl::OkStatus();
}

absl::Status LogitsProcessorChain::UpdateState(
    const ::litert::TensorBuffer& next_token_ids) {
  for (auto& processor : processors_) {
    LITERT_RETURN_IF_ERROR(processor->UpdateState(next_token_ids));
  }
  return absl::OkStatus();
}

absl::Status LogitsProcessorChain::UpdateState(absl::Span<int> next_token_ids) {
  for (auto& processor : processors_) {
    LITERT_RETURN_IF_ERROR(processor->UpdateState(next_token_ids));
  }
  return absl::OkStatus();
}

void LogitsProcessorChain::AddProcessor(
    std::unique_ptr<LogitsProcessor> processor) {
  if (processor) {
    processors_.push_back(std::move(processor));
  }
}

void LogitsProcessorChain::AddProcessor(
    std::unique_ptr<ConstrainedDecoder> processor) {
  if (processor) {
    constraint_decoder_ = processor.get();
    processors_.push_back(std::move(processor));
  }
}

bool LogitsProcessorChain::empty() const { return processors_.empty(); }

int LogitsProcessorChain::size() const { return processors_.size(); }

ConstrainedDecoder* LogitsProcessorChain::GetConstraintDecoder() const {
  return constraint_decoder_;
}

}  // namespace litert::lm
