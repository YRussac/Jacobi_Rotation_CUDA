#include <string>

class JacobiData {

public:
    JacobiData();

    JacobiData(int P, int d, float min_A, float max_A);

    void fill_indices_vectors();

    __device__
    void rotate(float *a, int i, int j, int k, int l, float c, float s);

    __device__
    void jacobi_product();

    void free_memory();

    void fill_angle_vectors();

//private:
    int d; // The matrix dimension
    int P; // The number of Jacobi matrices

    int *ip; // The first index of the different P Jacobi Rotations
    int *iq; // The second index of the different P Jacobi Rotations
    float *s; // Contains the sine values of the angle for the P rotations
    float *c; // Contains the cosine values of the angle for the P rotations
    float *theta;
    float *A; // Flattened array of the matrix A
};