#include "JacobiData.h"
#include "utils.h"

#include "cstdio"
#include "cstddef"
#include "cstdlib"
#include "cmath"
#include "random"
#include "iostream"

#include "boost/math/constants/constants.hpp"
float pi = (float) boost::math::constants::pi<long double>();

JacobiData::JacobiData() :
    JacobiData(10, 3, 0, 10){}

// Parameterized Constructor
JacobiData::JacobiData(int P, int d, int min_A, int max_A) {
    this->P = P;
    this->d = d;
    fill_indices_vectors();
    fill_angle_vectors();
    A = random_matrix(min_A, max_A, d);
}

void JacobiData::fill_indices_vectors() {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(0, d);

    // Allocate the memory
    ip = new int[P];
    iq = new int[P];
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

    c = new float[P];
    s = new float[P];
    theta = new float[P];

    for (int i = 0; i < P; ++i) {
        theta[i] = (float) dis(gen);
        c[i] = (float) cos(theta[i]);
        s[i] = (float) sin(theta[i]);
    }
}

void JacobiData::rotate(float **a, int i, int j, int k, int l, float c, float s) {
    float h;
    float g;
    g = a[i][j];
    h = a[k][l];
    a[i][j] = c * g - s * h;
    a[k][l] = s * g + c * h;
}

void JacobiData::free_memory() {
    // freeing the memory
    delete[] s;
    delete[] c;
    delete[] theta;
    delete[] ip;
    delete[] iq;
    free_matrix(A, d);
}

void JacobiData::jacobi_product() {

    print_matrix(A, d, "Initial matrix given");

    for (int step = 0; step < P; step++) {
        printf("Rotation n°%d out of %d \n", step, P - 1);
        printf("theta is equal to %f \n", theta[step]);
        printf("(ip,iq) = (%d,%d) \n", ip[step], iq[step]);
        for (int j = 0; j < d; j++) {
            rotate(A, ip[step], j, iq[step], j, c[step], s[step]);
        }
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                printf("%f\t", A[i][j]);
            }
            printf("\n");
        }
    }

    print_matrix(A, d, "Final matrix obtained");
}