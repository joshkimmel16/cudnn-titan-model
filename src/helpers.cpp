#include "helpers.h"

unsigned int number_tiles (unsigned int n_i, unsigned int n_n, unsigned int t_i, unsigned int t_n) {
    return (n_i / t_i) * (n_n / t_n);
}

unsigned int threads_per_tile (unsigned int t_i, unsigned int t_n) {
    return t_i * t_n;
}

unsigned int get_num_rounds (unsigned int num_tiles, unsigned int threads_tile, unsigned int num_sms, unsigned int max_sm) {
    unsigned int tmp1 = (max_sm % threads_tile == 0) ? (max_sm / threads_tile) : (max_sm / threads_tile) + 1; // how many tiles can be mapped to 1 SM
    unsigned int tmp2 = (num_tiles % tmp1 == 0) ? (num_tiles / tmp1) : (num_tiles / tmp1) + 1; // effective # of tiles that need to be mapped per round
    return (tmp2 % num_sms == 0) ? (tmp2 / num_sms) : (tmp2 / num_sms) + 1; // # of parallel rounds of execution
}

// note: this assumes all data required is already present in a register (i.e., ignores memory latency)
unsigned int vector_op (unsigned int len, unsigned int warp_size) {
    // assume atomic operations take 1 cycle
    // actual operation is broken out into atomic operations based on warp size relative to vector length
    unsigned int num_atomic = 1;
    return num_atomic * ((len % warp_size == 0) ? len/warp_size : (len/warp_size) + 1);
}

// note: this assumes all data required is already present in a register (i.e., ignores memory latency)
unsigned int tile_op (unsigned int len, unsigned int ht, unsigned elems_thread, unsigned int warp_size) {
    // per row of the weights matrix (ht)
    // must perform: vector-vector multiply, then a vector reduction, then an add to the output = 3 ops
    // # elements per thread effectively increases the length of the "vector"
    return ht * (3 * vector_op(len*elems_thread, warp_size));
}

unsigned int cycles_to_time(unsigned int cycles, unsigned int clock) {
    return cycles / clock; // because we are reporting in us
}