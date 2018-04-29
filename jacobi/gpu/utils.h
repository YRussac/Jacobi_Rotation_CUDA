#include <iostream>
#include <string>
#include "random"

int vec_idx(int row, int col, int d){
    return row * d + col;
}

void print_matrix(float* matrix, int d, std::string message){
    std::string line_break ("------------\n");

    std::cout << message << "\n";
    std::cout << line_break;

    for (int i = 0; i < d; i++) {
        for (int j = 0 ; j < d; j++) {
            std::cout << matrix[vec_idx(i, j, d)] << "\t";
        }
        std::cout << std::endl;
    }

}

float* f_vector(int vec_size){
//    float* vec = new float[vec_size];
    float* vec;
    cudaMallocManaged(&vec, vec_size*sizeof(float));
    return vec;
}

int* i_vector(int vec_size){
//    int* vec = new int[vec_size];
    int* vec;
    cudaMallocManaged(&vec, vec_size*sizeof(int));
    return vec;
}

float* random_array(int array_size, float min, float max){
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(min, max);

    float* a = f_vector(array_size);
    for (int i = 0; i< array_size; i++){
            a[i] = float(dis(gen));
    }
    return a;
}

void free_f_vec(float *vec){
    cudaFree(vec);
}

void free_i_vec(int *vec){
    cudaFree(vec);
}

