#include "utils.h"
#include "JacobiData.h"

#include "cstdio"
#include "cstddef"
#include "cstdlib"
#include "cmath"
#include "random"
#include "iostream"
#include "set"

#include "boost/math/constants/constants.hpp"
float pi = (float) boost::math::constants::pi<long double>();

JacobiData::JacobiData() :
    JacobiData(10, 3, 0, 10){}

// Parameterized Constructor
JacobiData::JacobiData(int P, int d, float min_A, float max_A) {
    this->P = P;
    this->d = d;
    fill_indices_vectors();
    fill_angle_vectors();
    A = random_array(d*d, min_A, max_A);
}

void JacobiData::fill_indices_vectors() {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(0, d);

    // Allocate the memory
    ip = i_vector(P);
    iq = i_vector(P);
    for (int i = 0; i < P; ++i) {
        int rand1 =  dis(gen);
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

__device__
void JacobiData::rotate(float* a, int i, int j, int k, float c, float s) {
    float h = a[vec_idx(i, j, d)];
    float g = a[vec_idx(k, j, d)];
    a[vec_idx(i, j, d)] = c * g - s * h;
    a[vec_idx(k, j, d)] = s * g + c * h;
}

void JacobiData::free_memory() {
    // freeing the memory
    free_f_vec(s);
    free_f_vec(c);
    free_f_vec(theta);
    free_i_vec(ip);
    free_i_vec(iq);
    free_f_vec(A);
}


__device__
void JacobiData::jacobi_product() {
    for (int step = 0; step < P; step++) {
        for (int j = 0; j < d; j++) {
            rotate(A, ip[step], j, iq[step], c[step], s[step]);
        }
    }
}

__device__
void JacobiData::jacobi_product_parallel_1(int block_size) {
//    Paallelise the matrix rotation only
    for (int step = 0; step < P; step++) {
//      Spread the columns rotations over the threads
        int index = threadIdx.x;
        int stride = block_size;
        for (int i = index; i < d; ++stride) {
            rotate(A, ip[step], i, iq[step], c[step], s[step]);
        }
        cudaDeviceSynchronize();
    }
}

__device__
int JacobiData::fetch_loop_range(int curr_idx){
    // TODO : debug and check
//    Returns the int of the index up to which the next product can performed in parallel
    std::set<int> idx_set;
    std::set<int>::iterator it;
    std::pair<std::set<int>::iterator,bool> ret;

    int i = curr_idx;
    bool stop = false;
    while (!stop) {
        const bool is_in_1 = idx_set.find(ip[i]) != idx_set.end();
        const bool is_in_2 = idx_set.find(iq[i]) != idx_set.end();
        i += 1;
        if (is_in_1 or is_in_2 or i == P){
            stop = true;
        }
    }
    return i;
}

__device__
void JacobiData::jacobi_product_parallel(int block_size) {
    // TODO : debug and check
//    Parallelise the matrix rotation only
    int curr_p_idx = 0;
    int curr_max_p = fetch_loop_range(curr_p_idx);

    while (curr_p_idx < P){
        int nb_p_mat = curr_max_p - curr_p_idx;
        for (int step = 0; step < nb_p_mat*d; step++) {
//      Spread the columns rotations over the threads
            int index = threadIdx.x;
            int stride = block_size;
            int col_idx;
            int mat_idx;

            for (int i = index; i < d; ++stride) {
                col_idx = i % d;
                mat_idx = (int) i / d;
                rotate(A, ip[mat_idx], i, iq[mat_idx], c[col_idx], s[col_idx]);
            }
        }
        cudaDeviceSynchronize();
    }
}