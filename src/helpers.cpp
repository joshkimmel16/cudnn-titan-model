#include "helpers.h"

unsigned int number_tiles (unsigned int n_i, unsigned int n_n, unsigned int t_i, unsigned int t_n) {
    return (n_i / t_i) * (n_n / t_n);
}

unsigned int threads_per_tile (unsigned int t_i, unsigned int t_n) {
    return t_i * t_n;
}

unsigned int get_num_rounds (unsigned int num_tiles, unsigned int threads_tile, TitanV m) {
    unsigned int tmp1 = m.max_threads_sm / threads_tile; // how many tiles can be mapped to 1 SM (want to round down in this case)
    unsigned int tmp2 = (num_tiles % tmp1 == 0) ? (num_tiles / tmp1) : (num_tiles / tmp1) + 1; // # of tiles normalized by mapping of multiple tiles to 1 SM
    return (tmp2 % m.num_sms == 0) ? (tmp2 / m.num_sms) : (tmp2 / m.num_sms) + 1; // # of parallel rounds of execution
}

// note: this assumes all data required is already present in a register (i.e., ignores memory latency)
// note: if the op is a memory op, it is assumed to take the same amount of time as other ops
unsigned int vector_op (unsigned int len, TitanV m) {
    // assume atomic operations take 1 cycle
    // actual operation is broken out into atomic operations based on warp size relative to vector length
    unsigned int num_atomic = 1;
    return num_atomic * ((len % m.warp_size == 0) ? len/m.warp_size : (len/m.warp_size) + 1);
}

// TODO: this only accounts for a single tile => must also account for the fact that all tiles are hitting the SAME L2
// TODO: this means an effective linear reduction in l2_bw based on # of concurrent tiles
unsigned int l2_latency(unsigned int num_accesses, TitanV m) {
    unsigned int data_amt = num_accesses * m.val_size; // compute total amount of data that will be transferred
    unsigned int t_transfer = data_amt / m.l2_bw; // determine time that will take based on L2-scratchpad BW
    return t_transfer * m.gpu_clock; // convert time to cycles using gpu_clock
}

// note: this assumes all data required is already present in registers (i.e., ignores memory latency)
unsigned int tile_op_1 (unsigned int len, unsigned int ht, unsigned elems_thread, TitanV m) {
    // per row of the weights matrix (ht)
    // must perform: vector-vector multiply, then a vector reduction, then an add to the output = 3 ops
    // # elements per thread effectively increases the length of the "vector"
    return ht * (3 * vector_op(len*elems_thread, m));
}

// note: this assumes all data required is already present in scratchpad (i.e., ignores memory latency)
unsigned int tile_op_2 (unsigned int len, unsigned int ht, unsigned elems_thread, TitanV m) {
    // per row of the weights matrix (ht)
    // must perform: vector-vector multiply, then a vector reduction, then an add to the output = 3 ops
    // # elements per thread effectively increases the length of the "vector"
    // must also load all vectors into vector registers and store result
    unsigned int inputs_reads = vector_op(len, m); // only need to read the input vector once
    unsigned int weights_reads = ht * inputs_reads; // must read (ht) weights vectors
    unsigned int work = ht * (3 * vector_op(len*elems_thread, m));
    unsigned int store = vector_op(ht, m); // only need to store the output vector once
    return inputs_reads + weights_reads + work + store;
}

// note: this assumes all data required is already present in L2 cache (i.e., ignores memory latency)
unsigned int tile_op_3 (unsigned int len, unsigned int ht, unsigned elems_thread, TitanV m) {
    // per row of the weights matrix (ht)
    // must perform: vector-vector multiply, then a vector reduction, then an add to the output = 3 ops
    // # elements per thread effectively increases the length of the "vector"
    // must also load all vectors into vector registers and store result
    // in addition, must account for latency of loads and stores from L2 => some of which is hidden by working on other ops
    unsigned int inputs_reads = vector_op(len, m); // only need to read the input vector once
    unsigned int weights_reads = ht * inputs_reads; // must read (ht) weights vectors
    unsigned int work = ht * (3 * vector_op(len*elems_thread, m));
    unsigned int store = vector_op(ht, m); // only need to store the output vector once

    // TODO: compute number of accesses, pass to l2_latency
    // TODO: determine "overlap" of L2 latency and processing work

    return inputs_reads + weights_reads + work + store;
}

unsigned int cycles_to_time(unsigned int cycles, TitanV m) {
    return cycles / m.gpu_clock; // because we are reporting in us
}