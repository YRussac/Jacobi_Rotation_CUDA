#include <stdio.h>
// Function that catches the error

void testCUDA(cudaError_t error, const char *file, int line){
  if (error != cudaSuccess){
    printf("Error in file %s at line %d \n", file , line);
    exit(EXIT_FAILURE);
  }
}

// Has to be define in the compilation in order to get the correct value of
// of the values __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__,__LINE__))

__global__ void empty_k(void){

}

int main (void){
  int count;
  cudaDeviceProp prop;

  empty_k<<<1,1>>>();
  testCUDA(cudaGetDeviceCount(&count));
  printf("The number of devices available is %i GPUs \n", count);
  testCUDA(cudaGetDeviceProperties(&prop, count-1));
  printf("Name %s\n", prop.name);
  printf("Global memory size in octet (bytes): %1d \n", prop.totalGlobalMem);
  printf("Shared memory size per block: %i\n", prop.sharedMemPerBlock);
  printf("Number of registers per block: %i\n", prop.regsPerBlock);
  printf("Number of threads in a warp: %i\n", prop.warpSize);
  printf("Maximum number of threads that can be launched per block: %i \n",
        prop.maxThreadsPerBlock);
  printf("Maximum number of threads that can be launched: %i X %i X %i\n",
        prop.maxThreadsDim[0],prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Maximum grid size: %i X %i X %i\n", prop.maxGridSize[0],
         prop.maxGridSize[1],prop.maxGridSize[2]);
  printf("Total Constant Memory Size: %1d\n", prop.totalConstMem);
  printf("Major Compute capability: %i\n", prop.major);
  printf("Minor Compute capability: %i\n", prop.minor);
  printf("Clock Rate : %i\n", prop.clockRate);
  printf("Maximum 1D texture memory: %i\n", prop.maxTexture1D);
  printf("Could we overlap? %i \n", prop.deviceOverlap);
  printf("Number of multiprocessors: %i \n", prop.multiProcessorCount);
  printf("Is there a limit for kernel execution? %i \n",
        prop.kernelExecTimeoutEnabled);
  printf("Is my GPU a chipset? %i\n", prop.integrated);
  printf("Can we map the host memory? %i \n", prop.canMapHostMemory);
  printf("Can we launch concurrent kernels: %i\n", prop.concurrentKernels);
  printf("Do we have ECC memory %i\n", prop.ECCEnabled);
  return 0;
}
