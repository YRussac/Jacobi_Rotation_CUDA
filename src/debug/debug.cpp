#include "cstdio"
#include "cstddef"
#include "cstdlib"
#include "cmath"
#include "random"
#include "iostream"
#include "set"
#include "algorithm"
#include "set.cpp"


int fetch_loop_range(int curr_idx, int *ip, int *iq, int P) {
//    Returns the int of the index up to which the next product can performed in parallel
    int i = curr_idx;
    bool stop = false;
    Set idx_set(2 * P);

    while (!stop) {
        const bool is_in_1 = idx_set.contains(ip[i]);
        const bool is_in_2 = idx_set.contains(iq[i]);
        if (is_in_1 or is_in_2 or i == P) {
            stop = true;
        }
        idx_set.insert(ip[i]);
        idx_set.insert(iq[i]);
        i += 1;
    }
    return i - 1;
}

int main() {
    int ip[3] = {1, 4, 5, 8};
    int iq[3] = {2, 3, 3, 5};
    int curr_idx = 0;
    int P = 4;

    int next_idx = fetch_loop_range(curr_idx, ip, iq, P);
    std::cout << next_idx << std::endl;

    next_idx = fetch_loop_range(curr_idx, ip, iq, P);
    std::cout << next_idx << std::endl;

    next_idx = fetch_loop_range(curr_idx, ip, iq, P);
    std::cout << next_idx << std::endl;
    return 0;
}