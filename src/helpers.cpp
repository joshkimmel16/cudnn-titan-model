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
    // actual operation is broken out into vector operations based on warp size relative to vector length
    return m.cpi * ((len % m.warp_size == 0) ? len/m.warp_size : (len/m.warp_size) + 1);
}

// obtain greatest advtange when working on very many threads in parallel
double latency_hide(unsigned int num_threads, TitanV m) {
    double adv = num_threads / m.max_threads_sm;
    return 1 - m.max_lat_hide * adv;
}

// must account for the fact that multiple blocks can be trying to accumulate to the same output memory location
// if so, these operations must be synchronized s.t. the end result is correct
// synchronization penalty depends on # of thread blocks that must synchronize
unsigned int sync_latency(unsigned int num_sync, TitanV m) {
    return num_sync * m.sync_penalty; 
}

// note: this assumes all data required is already present in registers (i.e., ignores memory latency)
unsigned int tile_op_1 (unsigned int len, unsigned int ht, unsigned elems_thread, TitanV m) {
    // per row of the weights matrix (ht)
    // must perform: vector-vector multiply, then a vector reduction, then an add to the output
    // assumed that the reduction takes 5 cycles (log(32)) => total = 7 cycles
    // # elements per thread effectively increases the length of the "vector"
    return ht * (7 * vector_op(len*elems_thread, m));
}

// note: this assumes all data required is already present in scratchpad (i.e., ignores memory latency)
unsigned int tile_op_2 (unsigned int len, unsigned int ht, unsigned elems_thread, TitanV m) {
    // per row of the weights matrix (ht)
    // must perform: vector-vector multiply, then a vector reduction, then an add to the output
    // assumed that the reduction takes 5 cycles (log(32)) => total = 7 cycles
    // # elements per thread effectively increases the length of the "vector"
    // must also load all vectors into vector registers and store result
    unsigned int inputs_reads = vector_op(len, m); // only need to read the input vector once
    unsigned int weights_reads = ht * inputs_reads; // must read (ht) weights vectors
    unsigned int work = ht * (7 * vector_op(len*elems_thread, m));
    unsigned int store = vector_op(ht, m); // only need to store the output vector once
    return inputs_reads + weights_reads + work + store;
}

// note: this assumes all data required is already present in L2 cache (i.e., ignores memory latency)
unsigned int tile_op_3 (unsigned int len, unsigned int ht, unsigned elems_thread, unsigned int tiles_round, unsigned int tiles_sm, unsigned int tile_overlap, TitanV m) {
    // per row of the weights matrix (ht)
    // must perform: vector-vector multiply, then a vector reduction, then an add to the output
    // assumed that the reduction takes 5 cycles (log(32)) => total = 7 cycles
    // # elements per thread effectively increases the length of the "vector"
    // must also load all vectors into vector registers and store result
    // in addition, must account for latency of loads and stores from L2 => some of which is hidden by working on other ops
    unsigned int inputs_reads = vector_op(len, m); // only need to read the input vector once
    unsigned int weights_reads = ht * inputs_reads; // must read (ht) weights vectors
    unsigned int work = ht * (7 * vector_op(len*elems_thread, m));
    unsigned int store = vector_op(ht, m); // only need to store the output vector once
    unsigned int sync = sync_latency(tile_overlap, m); // compute sync penalty

    // determine "overlap" of L2 latency and processing work
    // this is driven by: nominal # of actions => load a bunch, start working, thread in subsequent loads strategically
    unsigned int l2_lat_obs = (unsigned int)((double)m.l2_lat * latency_hide((tiles_sm*len*ht/elems_thread), m));
    
    return inputs_reads + weights_reads + work + store + sync + l2_lat_obs;
}

// Factors in memory latency by estimating a proportion of the data that could already be in L2 and the rest that must be fetched from memory
unsigned int tile_op_4 (unsigned int len, unsigned int ht, unsigned elems_thread, unsigned int tiles_round, unsigned int tiles_sm, unsigned int tile_overlap, TitanV m) {
    // per row of the weights matrix (ht)
    // must perform: vector-vector multiply, then a vector reduction, then an add to the output
    // assumed that the reduction takes 5 cycles (log(32)) => total = 7 cycles
    // # elements per thread effectively increases the length of the "vector"
    // must also load all vectors into vector registers and store result
    // in addition, must account for latency of loads and stores from L2 and memory => some of which is hidden by working on other ops
    unsigned int inputs_reads = vector_op(len, m); // only need to read the input vector once
    unsigned int weights_reads = ht * inputs_reads; // must read (ht) weights vectors
    unsigned int work = ht * (7 * vector_op(len*elems_thread, m));
    unsigned int store = vector_op(ht, m); // only need to store the output vector once
    unsigned int sync = sync_latency(tile_overlap, m); // compute sync penalty
    unsigned int num_access = (inputs_reads + weights_reads + store) / m.cpi; // must normalize based on CPI

    // proportion of reqested data that can fit in the l2 cache
    double p = (double) m.l2_cap/(num_access * m.warp_size * m.val_size);

    // if all data can fit in cache than only l2_latency
    // else access p data with cost l2_lat and 1 - p data with l2_lat + mem_lat cost
    double total_lat = (p >= 1) ? (double) m.l2_lat : (double)(p * m.l2_lat + (1 - p) * (m.l2_lat + m.mem_lat));

    // determine "overlap" of L2 latency and processing work
    // this is driven by: nominal # of actions => load a bunch, start working, thread in subsequent loads strategically
    unsigned int lat_obs = (unsigned int)(total_lat * latency_hide((tiles_sm * len * ht/elems_thread), m));
    return inputs_reads + weights_reads + work + store + sync + lat_obs;
}
	
unsigned int cycles_to_time(unsigned int cycles, TitanV m) {
    return cycles / m.gpu_clock; // because we are reporting in us
}
