#pragma once

// CUDA runtime
#include "cuda_runtime.h"
#include "gpu/helper_cuda.h"

int matrixMultiply(int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB);
