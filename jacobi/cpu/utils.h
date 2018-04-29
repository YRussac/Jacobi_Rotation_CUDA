#include <iostream>
#include <string>
#include "random"

void print_matrix(float** matrix, int d, std::string message){
    std::string line_break ("------------\n");

    std::cout << message << "\n";
    std::cout << line_break;

    for (int i = 0; i < d; i++) {
        for (int j = 0 ; j < d; j++) {
            std::cout << matrix[i][j] << "\t";
        }
        std::cout << std::endl;
    }

}

float* f_vector(int vec_size){
    float* vec = new float[vec_size];
    return vec;
}

int* i_vector(int vec_size){
    int* vec = new int[vec_size];
    return vec;
}

float** matrix(int nrl, int nrh, int ncl, int nch)
/* allocate a float matrix with subscript range m[nrl...nrh][ncl...nch] */
{
    int n_row = nrh-nrl+1;
    int n_col = nch-ncl+1;
    float **m;

    m = new float*[n_row];
    for (int i = 0; i < n_col; ++i) {
        m[i] = new float[n_col];
    }

    return m;
}

float** random_matrix(int min, int max, int d){
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(min, max);

    float** a = matrix(0, d, 0, d);
    for (int i = 0; i< d; i++){
        for (int j=0; j<d; j++){
            a[i][j] = float(dis(gen));
        }
    }
    return a;
}

void free_matrix(float **m, int n_col)
/* free a float matrix allocated by matrix() */
{
    for (int i = 0; i < n_col; ++i)
        delete m[i];
    delete[] m;
}
