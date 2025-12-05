#include <math.h>

// --- Baseline Kernel (Naive, Unoptimized) ---
// This uses a direct 1-to-1 mapping from thread index to array index.

__global__ void baseline_elementwise_kernel(float *output, const float *input,
                                            int N) {

  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId >= N)
    return;

  int i_minus_1 = threadId - 1;
  int i_plus_1 = threadId + 1;

  float sum = input[threadId];

  if (i_minus_1 >= 0) {
    sum += input[i_minus_1];
  }

  if (i_plus_1 < N) {
    sum += input[i_plus_1];
  }

  output[threadId] = sum;
}