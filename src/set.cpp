#include "constants.h"

class Set {
    int size;
    int* elements;
    int curr_idx = 0;

public:
    Set();
    Set(int size);
    void insert(int);
    bool contains(int);
    void reset();
};

Set::Set(): Set(2 * GLOBAL_P) {}

Set::Set(int size){
    this->size = size;
    this->elements = i_vector(size);
}

__device__ __host__
void Set::insert(int element) {
    elements[curr_idx] = element;
    curr_idx += 1;
}

__device__ __host__
bool Set::contains(int element) {
    bool exists = false;
    for (int i = 0; i < curr_idx; ++i) {
        if (elements[i]  == element){
            exists = true;
        }
    }
    return exists;
}

__device__ __host__
void Set::reset() {
    curr_idx = 0;
}
