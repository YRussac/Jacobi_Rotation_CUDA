#include "JacobiData.cu"
#include "random"
#include "iostream"
#include <ctime>


void testCUDA(cudaError_t error, const char *file, int line){
    if (error != cudaSuccess){
        printf("Error in file %s at line %d \n", file , line);
        exit(EXIT_FAILURE);
    }
}

// Has to be define in the compilation in order to get the correct value of
// of the values __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__,__LINE__))


__global__ void compute(JacobiData *jacobi_array, int optimisation) {

        int index = blockIdx.x;
        int stride = N_BLOCKS;
        for (int i = index; i < N_PROBLEMS; i += stride) {
            if (optimisation == 1){
                jacobi_array[i].jacobi_product();
            }
            if (optimisation == 2){
                jacobi_array[i].jacobi_product_parallel_cols(N_THREADS);
            }
            if (optimisation == 3) {
                jacobi_array[i].jacobi_product_parallel(N_THREADS);
            }
        }
    }



float cpu_run(JacobiData *jacobi_array) {
    int count;
    cudaDeviceProp prop;
    testCUDA(cudaGetDeviceCount(&count));
    testCUDA(cudaGetDeviceProperties(&prop, count-1));
    float TimerV;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start,0));

    cudaMallocManaged(&jacobi_array, N_PROBLEMS * sizeof(JacobiData));
    for (int j = 0; j < N_PROBLEMS; ++j) {
        jacobi_array[j] = JacobiData();
        if(DEBUG){
            jacobi_array[j].debug_fill();
        }
    }
    if(DEBUG){
        print_matrix(jacobi_array[0].A, jacobi_array[0].d, "Initial matrix A number 0");
    }

    for (int i = 0; i < N_PROBLEMS; ++i) {
        jacobi_array[i].jacobi_product();
    }

    if(DEBUG){
        print_matrix(jacobi_array[0].A, jacobi_array[0].d, "Output matrix A number 0");
    }

    for (int i = 0; i < N_PROBLEMS; ++i) {
        jacobi_array[i].free_memory();
    }


    testCUDA(cudaEventRecord(stop,0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerV,start,stop));
    return TimerV;
}

float gpu_run(JacobiData *jacobi_array, int optimisation) {
    int count;
    cudaDeviceProp prop;
    testCUDA(cudaGetDeviceCount(&count));
    testCUDA(cudaGetDeviceProperties(&prop, count-1));
    float TimerV;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start,0));

    cudaMallocManaged(&jacobi_array, N_PROBLEMS * sizeof(JacobiData));
    for (int j = 0; j < N_PROBLEMS; ++j) {
        jacobi_array[j] = JacobiData();
        if (DEBUG) {
            jacobi_array[j].debug_fill();
        }
    }

    if(DEBUG){
        print_matrix(jacobi_array[0].A, jacobi_array[0].d, "Initial matrix A number 0");
    }

//    On GPU
    compute << < N_BLOCKS, N_THREADS >> > (jacobi_array, optimisation);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    if(DEBUG){
        print_matrix(jacobi_array[0].A, jacobi_array[0].d, "Output matrix A number 0");
    }


    for (int i = 0; i < N_PROBLEMS; ++i) {
        jacobi_array[i].free_memory();
    }

    testCUDA(cudaEventRecord(stop,0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerV,start,stop));
//    printf("Execution time: %f ms\n", TimerV);
    return TimerV;
}

int main() {
//  Define the array of problems
    JacobiData *jacobi_array;

    float duration = 0.;
    for (int i = 0; i < NB_EXP; ++i) {
        duration += cpu_run(jacobi_array);
    }
    duration /= NB_EXP;
    printf("Execution time CPU : %f ms\n", duration);

    duration = 0.;
    for (int i = 0; i < NB_EXP; ++i) {
        duration += gpu_run(jacobi_array, OPTIMISATION);
    }
    duration /= NB_EXP;
    printf("Execution time GPU : %f ms\n", duration);

    return 0;
}