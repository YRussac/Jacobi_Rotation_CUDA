#include "JacobiData.cpp"
#include "random"
#include "iostream"

int main(){


    struct jacobi {
        int* ip;
    } ;

    jacobi.ip = new int[5];
    for (int i = 0; i < 5; ++i) {
        jacobi.ip[i] = i;
    }
    return 0;

}