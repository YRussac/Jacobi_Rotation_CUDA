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


#define testCUDA(error) (testCUDA(error, __FILE__,__LINE__))

__global__ void compute(JacobiData *jacobi_array, int optimisation) {


    if (optimisation == 0){
        for (int i = 0; i < N_PROBLEMS; ++i) {
            jacobi_array[i].jacobi_product();
        }
    }
    else{
        int block_index = blockIdx.x;
        int block_stride = N_BLOCKS;
        for (int i_block = block_index; i_block < N_PROBLEMS; i_block += block_stride) {


                if (optimisation == 2){
                    //    Parallelise the matrix rotation only
                    int th_index = threadIdx.x;
                    int th_stride = blockDim.x;
                    for (int th_step = 0; th_step < jacobi_array[i_block].P; th_step++) {
                    // Spread the columns rotations over the threads
                        for (int i_th = th_index; i_th < jacobi_array[i_block].d; i_th += th_stride) {
                            jacobi_array[i_block].abstract_rotate(th_step, i_th);
                        }
                        //synchronize the local threads in the block
                        __syncthreads();
                    }

//                    jacobi_array[i_block].jacobi_product_parallel_cols(N_THREADS - 1);
                }
                if (optimisation == 3) {
                    int col_idx;
                    int mat_idx;
                    int P = jacobi_array[i_block].P;
                    int d = jacobi_array[i_block].d;

                    int curr_p_idx = 0;
                    int curr_max_p = jacobi_array[i_block].fetch_loop_range(curr_p_idx);

                    while (curr_p_idx < P){
                        int nb_p_mat = curr_max_p - curr_p_idx;
                        int th_index = threadIdx.x;
                        int th_stride = blockDim.x;

                        for (int i_th = th_index; i_th < nb_p_mat * d; i_th+=th_stride) {
                            col_idx = i_th % d;
                            mat_idx = curr_p_idx + (int) i_th / d;
                            jacobi_array[i_block].abstract_rotate(mat_idx, col_idx);
//                            rotate(A, ip[mat_idx], col_idx, iq[mat_idx], c[mat_idx], s[mat_idx]);
                        }
                        //synchronize the local threads in the block
                        __syncthreads();
                        curr_p_idx = curr_max_p;
                        curr_max_p = jacobi_array[i_block].fetch_loop_range(curr_p_idx);

                    }
                }

                if (optimisation == 1){
                    if (threadIdx.x == (N_THREADS - 1)){
                        jacobi_array[i_block].jacobi_product();
                }

            }
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

    int duration;
    for (int i = 0; i < N_PROBLEMS; ++i) {
        clock_t start_time = clock();
        jacobi_array[i].jacobi_product();
        clock_t stop_time = clock();
        duration = (int)(stop_time - start_time);
//        printf("Hello from time %d\n",
//               duration);
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
//    float duration = 0.;
//    for (int i = 0; i < NB_EXP; ++i) {
//        duration += cpu_run(jacobi_array);
//    }
//    duration /= NB_EXP;
//    printf("Execution time CPU : %f ms\n", duration);
//
//    printf("\n------------------------------------------\n");

    float duration = 0.;
    for (int i = 0; i < NB_EXP; ++i) {
        duration += gpu_run(jacobi_array, OPTIMISATION);
    }
    duration /= NB_EXP;
    printf("Execution time GPU %i : %f ms\n",OPTIMISATION, duration);

    return 0;
}