#include "JacobiData.cu"
#include "random"
#include "iostream"


void testCUDA(cudaError_t error, const char *file, int line){
    if (error != cudaSuccess){
        printf("Error in file %s at line %d \n", file , line);
        exit(EXIT_FAILURE);
    }
}

#define testCUDA(error) (testCUDA(error, __FILE__,__LINE__))

__global__ void compute(JacobiData *jacobi_array, int optimisation) {
//    int block_size = blockDim.x;
//    switch (optimisation){
//        case 1 : jacobi_array[blockIdx.x].jacobi_product();
//            break;
//        case 2 : jacobi_array[blockIdx.x].jacobi_product_parallel_cols(N_THREADS);
//            break;
//        case 3 : jacobi_array[blockIdx.x].jacobi_product_parallel(N_THREADS);
//            break;
//    }
    jacobi_array[blockIdx.x].jacobi_product();
}


void cpu_run(JacobiData *jacobi_array) {


    int count;
    cudaDeviceProp prop;
    testCUDA(cudaGetDeviceCount(&count));
    testCUDA(cudaGetDeviceProperties(&prop, count-1));
    float TimerV;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start,0));

    for (int i = 0; i < N_PROBLEMS; ++i) {
        jacobi_array[i].jacobi_product();
    }

    testCUDA(cudaEventRecord(stop,0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerV,start,stop));
    printf("Execution time: %f ms\n", TimerV);

}

void gpu_run(JacobiData *jacobi_array, int optimisation) {
    int count;
    cudaDeviceProp prop;
    testCUDA(cudaGetDeviceCount(&count));
    testCUDA(cudaGetDeviceProperties(&prop, count-1));
    float TimerV;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start,0));

//    print_matrix(jacobi_array[0].A, jacobi_array[0].d, "Initial matrix A number 0");

//    On GPU
    compute << < N_BLOCKS, N_THREADS >> > (jacobi_array, optimisation);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

//    print_matrix(jacobi_array[0].A, jacobi_array[0].d, "Output matrix A number 0");


    testCUDA(cudaEventRecord(stop,0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerV,start,stop));
    printf("Execution time: %f ms\n", TimerV);


}

int main() {
    int optimisation = 0;

    //  Define the array of problems
    JacobiData *jacobi_array;
    cudaMallocManaged(&jacobi_array, N_PROBLEMS * sizeof(JacobiData));
    for (int j = 0; j < N_PROBLEMS; ++j) {
        jacobi_array[j] = JacobiData();
//        jacobi_array[j].debug_fill();
    }

    cpu_run(jacobi_array);
    gpu_run(jacobi_array, 1);

//    switch(optimisation) {
//            case 0 : cpu_run(jacobi_array);
//            case 1 : gpu_run(jacobi_array, 1);
//            break;
//            case 2 : gpu_run(jacobi_array, 2);
//            break;
//            case 3 : gpu_run(jacobi_array, 3);
//            break;
//        }


    for (int i = 0; i < N_PROBLEMS; ++i) {
        jacobi_array[i].free_memory();
    }

    return 0;

}