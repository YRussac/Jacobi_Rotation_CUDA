#include "JacobiData.cu"
#include "random"
#include "iostream"


__global__ void compute(JacobiData *jacobi_array, int optimisation) {
    int block_size = blockDim.x;
    if (optimisation == 1){
        jacobi_array[blockIdx.x].jacobi_product_parallel_cols(block_size);
    }
    if (optimisation == 2){
        jacobi_array[blockIdx.x].jacobi_product_parallel_cols(block_size);
    }
    if (optimisation == 3) {
        jacobi_array[blockIdx.x].jacobi_product_parallel(block_size);
    }

}


void cpu_run(JacobiData *jacobi_array) {
    if(DEBUG){
        for (int i = 0; i < N_PROBLEMS; ++i) {
            jacobi_array[i].debug_fill();
        }

    }
    for (int i = 0; i < N_PROBLEMS; ++i) {
        jacobi_array[i].jacobi_product();
    }
}

void gpu_run(JacobiData *jacobi_array, int optimisation) {

//    On GPU
    compute << < N_BLOCKS, N_THREADS >> > (jacobi_array, optimisation);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();


}

int main() {
//  Define the array of problems
    JacobiData *jacobi_array;
    cudaMallocManaged(&jacobi_array, N_PROBLEMS * sizeof(JacobiData));
    for (int j = 0; j < N_PROBLEMS; ++j) {
        jacobi_array[j] = JacobiData();
        jacobi_array[j].debug_fill();
    }

    if(DEBUG){
        print_matrix(jacobi_array[0].A, jacobi_array[0].d, "Initial matrix A number 0");
    }

    if(OPTIMISATION == 0){
        cpu_run();
    }
    else{
        gpu_run(jacobi_array, OPTIMISATION);
    }


    for (int i = 0; i < N_PROBLEMS; ++i) {
        jacobi_array[i].free_memory();
    }

    if(DEBUG){
        print_matrix(jacobi_array[0].A, jacobi_array[0].d, "Output matrix A number 0");
    }

    return 0;



}