# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//:__subpackages__",
    ],
)

licenses(["notice"])

cc_library(
    name = "logits_processor",
    hdrs = ["logits_processor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//tflite/types:half",
    ],
)

cc_library(
    name = "logits_processor_chain",
    srcs = ["logits_processor_chain.cc"],
    hdrs = ["logits_processor_chain.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":logits_processor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/components/logits_processor/constrained_decoding:constrained_decoder",
        "//runtime/util:litert_status_util",
        "@litert//tflite/types:half",
    ],
)

cc_test(
    name = "logits_processor_chain_test",
    srcs = ["logits_processor_chain_test.cc"],
    deps = [
        ":logits_processor",
        ":logits_processor_chain",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/components/logits_processor/constrained_decoding:constrained_decoder",
        "//runtime/components/logits_processor/constrained_decoding:fake_constraint",
        "@litert//tflite/types:half",
    ],
)
