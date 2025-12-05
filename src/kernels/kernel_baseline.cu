#include <math.h>

// --- Baseline Kernel (Naive, Unoptimized) ---
// This uses a direct 1-to-1 mapping from thread index to array index.

__global__ void baseline_elementwise_kernel(float *output, const float *input,
                                            int N) {

  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId >= N)
    return;

  if (threadId == N) {
    output[threadId] = input[threadId] + input[threadId + 1];
  } else if (threadId == 0) {
    output[threadId] = input[threadId] + input[threadId + 1];
  } else {
    output[threadId] =
        input[threadId - 1] + input[threadId] + input[threadId + 1];
  }
}