#include <string>
#include "set.cpp"

class JacobiData {

public:
    JacobiData();

    JacobiData(int P, int d, float min_A, float max_A, int n_threads);

    void fill_indices_vectors();

    void rotate(float *a, int i, int j, int k, float c, float s);

    void jacobi_product();

    void abstract_rotate(int step, int i);

    void free_memory();

    void fill_angle_vectors();

    void debug_fill();

    void jacobi_product_parallel_cols(int block_size);

    void jacobi_product_parallel(int block_size);

    int fetch_loop_range(int curr_idx);

    //private:
    int d; // The matrix dimension
    int P; // The number of Jacobi matrices

    int *ip; // The first index of the different P Jacobi Rotations
    int *iq; // The second index of the different P Jacobi Rotations
    float *s; // Contains the sine values of the angle for the P rotations
    float *c; // Contains the cosine values of the angle for the P rotations
    float *theta;
    float *A;// Flattened array of the matrix A
//    Set* sets; // Sets used for the parallelisation
    Set sets; // Sets used for the parallelisation
};