#include "utils.h"
#include "JacobiData.h"

#include "cstdio"
#include "cstddef"
#include "cstdlib"
#include "cmath"
#include "random"
#include "iostream"
#include "set"
#include "algorithm"

#include "boost/math/constants/constants.hpp"


float pi = (float) boost::math::constants::pi<long double>();

JacobiData::JacobiData() :
        JacobiData(GLOBAL_P, GLOBAL_d, GLOBAL_min_A, GLOBAL_max_A, N_THREADS) {}

// Parameterized Constructor
JacobiData::JacobiData(int P, int d, float min_A, float max_A, int n_threads) {
    this->P = P;
    this->d = d;
    fill_indices_vectors();
    fill_angle_vectors();
    A = random_array(d * d, min_A, max_A);

//    Allocate sets for parallel computing
    cudaMallocManaged(&sets, n_threads * sizeof(Set));
    for (int j = 0; j < n_threads; ++j) {
        sets[j] = Set();
    }
}

void JacobiData::fill_indices_vectors() {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(0, d);

    // Allocate the memory
    ip = i_vector(P);
    iq = i_vector(P);
    for (int i = 0; i < P; ++i) {
        int rand1 = dis(gen);
        int rand2 = rand1;
        while (rand2 == rand1) {
            rand2 = dis(gen);
        }
        ip[i] = rand1;
        iq[i] = rand2;
    }
}

void JacobiData::fill_angle_vectors() {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0., 2. * pi);

    c = f_vector(P);
    s = f_vector(P);
    theta = f_vector(P);

    for (int i = 0; i < P; ++i) {
        theta[i] = (float) dis(gen);
        c[i] = (float) cos(theta[i]);
        s[i] = (float) sin(theta[i]);
    }
}

void JacobiData::free_memory() {
    // freeing the memory
    free_f_vec(s);
    free_f_vec(c);
    free_f_vec(theta);
    free_i_vec(ip);
    free_i_vec(iq);
    free_f_vec(A);
// free other tables
}

void JacobiData::debug_fill() {
    int new_arr_1[] = {1, 2, 0};
    std::copy(new_arr_1, new_arr_1 + 3, ip);

    int new_arr_2[] = {0, 1, 2};
    std::copy(new_arr_2, new_arr_2 + 3, iq);

    float new_arr_3[] = {0, pi / 4, 2 * pi / 3};
    std::copy(new_arr_3, new_arr_3 + 3, theta);

    for (int i = 0; i < 3; ++i) {
        c[i] = (float) cos(theta[i]);
        s[i] = (float) sin(theta[i]);
    }

    float new_arr_4[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    std::copy(new_arr_4, new_arr_4 + 9, A);
}

__device__ __host__

void JacobiData::jacobi_product() {
    for (int step = 0; step < P; step++) {
        for (int j = 0; j < d; j++) {
            rotate(A, ip[step], j, iq[step], c[step], s[step]);
        }
    }
}

__device__ __host__

void JacobiData::rotate(float *a, int i, int j, int k, float c, float s) {
    if (i > k) {
        swap(&i, &k);
    }
    float h = a[vec_idx(i, j, d)];
    float g = a[vec_idx(k, j, d)];
    a[vec_idx(i, j, d)] = s * g + c * h;
    a[vec_idx(k, j, d)] = c * g - s * h;
}

__device__
void JacobiData::jacobi_product_parallel_cols(int block_size) {
//    Parallelise the matrix rotation only
    int index = threadIdx.x;
    int stride = block_size;
    for (int step = 0; step < P; step++) {
// Spread the columns rotations over the threads
        for (int i = index; i < d; i += stride) {
            rotate(A, ip[step], i, iq[step], c[step], s[step]);
        }
        //synchronize the local threads in the block
        __syncthreads();
    }
}

__device__
int JacobiData::fetch_loop_range(int curr_idx) {
//    Returns the int of the index up to which the next product can performed in parallel
    int i = curr_idx;
    bool stop = false;
    int index = threadIdx.x;

    while (!stop) {
        const bool is_in_1 = sets[index].contains(ip[i]);
        const bool is_in_2 = sets[index].contains(iq[i]);
        if (is_in_1 or is_in_2 or i == P) {
            stop = true;
        }
        sets[index].insert(ip[i]);
        sets[index].insert(iq[i]);
        i += 1;
    }
    sets[index].reset();
    return i - 1;
}


__device__
void JacobiData::jacobi_product_parallel(int block_size) {
    // TODO : debug and check
    int col_idx;
    int mat_idx;

    int curr_p_idx = 0;
    int curr_max_p = fetch_loop_range(curr_p_idx);

    while (curr_p_idx < P){
    int nb_p_mat = curr_max_p - curr_p_idx;
    int index = threadIdx.x;
    int stride = block_size;
//
    int i = index;
        for (int i = index; i < nb_p_mat * d; i+=stride) {
            col_idx = i % d;
            mat_idx = (int) i / d + curr_p_idx;
            rotate(A, ip[mat_idx], col_idx, iq[mat_idx], c[mat_idx], s[mat_idx]);
        }
    //synchronize the local threads in the block
    __syncthreads();
    curr_p_idx = curr_max_p;
    curr_max_p = fetch_loop_range(curr_p_idx);

    }
}