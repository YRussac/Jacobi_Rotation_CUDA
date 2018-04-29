#include "JacobiData.cpp"
#include "random"
#include "iostream"

int main(){
//    int P = 1000;
//    int d = 3;
//    int min_A = 0;
//    int max_A = 10;
//    JacobiData Jacobi(P, d, min_A, max_A);
//    Jacobi.jacobi_product();
//
//    std::random_device rd;  //Will be used to obtain a seed for the random number engine
//    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
//    std::uniform_real_distribution<> dis(0., 2. * float(pi));

    JacobiData jacobi_array[5];

    jacobi_array[0].jacobi_product();

    return 0;

}