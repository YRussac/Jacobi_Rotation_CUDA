#include "JacobiData.cu"
#include "random"
#include "iostream"
//#include <boost>

//namespace po = boost::program_options;


__global__ void compute(JacobiData *jacobi_array, int optimisation) {
    int block_size = blockDim.x;
    switch (optimisation){
        case 1 : jacobi_array[blockIdx.x].jacobi_product_parallel_cols(block_size);
        case 2 : jacobi_array[blockIdx.x].jacobi_product_parallel(block_size);
    }
}


void cpu_run() {
    JacobiData jacobi;
    jacobi.debug_fill();
    jacobi.jacobi_product();
    jacobi.free_memory();
}

void gpu_run(int optimisation) {
    const int n_problems = 1;
    const int n_blocks = n_problems;

//  Define the array of problems
    JacobiData *jacobi_array;
    cudaMallocManaged(&jacobi_array, n_problems * sizeof(JacobiData));
    for (int j = 0; j < n_problems; ++j) {
        jacobi_array[j] = JacobiData();
        jacobi_array[j].debug_fill();
    }

    print_matrix(jacobi_array[0].A, jacobi_array[0].d, "Initial matrix A number 0");

//    On GPU
    compute << < n_blocks, N_THREADS >> > (jacobi_array, optimisation);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    print_matrix(jacobi_array[0].A, jacobi_array[0].d, "Output matrix A number 0");

    for (int i = 0; i < n_problems; ++i) {
        jacobi_array[i].free_memory();
    }
}

int main() {
    // Declare the supported options.
//    po::options_description desc("Required options");
//    desc.add_options()
//            ("help", "The option is required")
//            ("optimisation", po::value<int>(),
//                    "Set the optimisation level \n"
//                    " 0 : Run on CPU \n"
//                    " 1 : Parallelize over columns only \n"
//                    " 2 : Parallelize over columns and matrices")
//            ;
//    po::variables_map vm;
//    po::store(po::parse_command_line(ac, av, desc), vm);
//    po::notify(vm);

//    if (vm.count("optimisation")) {
//        int optimisation = vm["compression"].as<int>();
    int optimisation = 1;
    switch(optimisation) {
        case 0 : cpu_run();
        break;
        case 1 : gpu_run(1);
            break;
        case 2 : gpu_run(2);
            break;
    }
//    } else {
//        cout << "Optimisation level was not set.\n";
//    }
    return 0;

}