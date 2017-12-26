package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "matrix_mult_cc",
    srcs = [
        "matrix_mult.cpp",
        "matrix_mult_gpu_test.cu.cpp",
        "helper_cuda.h",
        "helper_string.h",
        "matrix_mult_gpu_test.h"
    ],
    deps = [
        "@local_config_cuda//cuda:cuda",
    ],
    copts = [
        "-x",
        "cuda",
        "-DGOOGLE_CUDA=1",
        "-DCUDA_FOUND"
    ],
)
