#include <cmath>
#include <curand_kernel.h>
#include <iostream>

const int THREADS_PER_BLOCK = 256;
const int NUM_BLOCKS = 256;
const int TOTAL_THREADS = THREADS_PER_BLOCK * NUM_BLOCKS;
const int NUM_POINTS =
    TOTAL_THREADS * 15260; // Total number of points to generate (around 1_000_000_000)

__global__ void monteCarloPi(float *d_result, curandState *d_states,
                             int numPoints) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int totalThreads = blockDim.x * gridDim.x;

  int pointsPerThread = numPoints / totalThreads;

  curandState localState = d_states[tid];
  int pointsInsideCircle = 0;

  for (int i = 0; i < pointsPerThread; ++i) {
    float x = curand_uniform(&localState);
    float y = curand_uniform(&localState);

    if (x * x + y * y <= 1.0f + 1e-6) {
      pointsInsideCircle++;
    }
  }

  d_states[tid] = localState;

  atomicAdd(d_result, static_cast<float>(pointsInsideCircle));
}

__global__ void initRandomStates(curandState *states, unsigned long long seed) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, tid, 0, &states[tid]);
}

int main() {
  float *h_result;
  float *d_result;
  cudaError_t cudaStatus = cudaMalloc((void **)&d_result, sizeof(float));
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus)
              << std::endl;
    return 1;
  }
  curandState *d_states;

  h_result = new float[1];
  h_result[0] = 0;

  cudaMalloc((void **)&d_result, sizeof(float));
  cudaMalloc((void **)&d_states, TOTAL_THREADS * sizeof(curandState));

  cudaMemcpy(d_result, h_result, sizeof(float), cudaMemcpyHostToDevice);

  // Initialize random states
  initRandomStates<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_states, time(nullptr));

  monteCarloPi<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_result, d_states,
                                                  NUM_POINTS);

  cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

  // Calculate Pi using the Monte Carlo method
  float pi = 4.0f * h_result[0] / static_cast<float>(NUM_POINTS);
  std::cout << "Estimated Pi: " << pi << std::endl;

  // Clean up
  delete[] h_result;
  cudaFree(d_result);
  cudaFree(d_states);

  return 0;
}
%