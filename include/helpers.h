#ifndef HELPERS_H
#define HELPERS_H

// compute the theoretical # of tiles necessary given the provided input parameters:
// input/output dimensions, tile dimensions
unsigned int number_tiles (unsigned int n_i, unsigned int n_n, unsigned int t_i, unsigned int t_n);

// compute the theoretical # of tiles necessary given the provided input parameters:
// tile dimensions
unsigned int threads_per_tile (unsigned int t_i, unsigned int t_n);

// compute the # of sequences of tile executions that are needed given:
// total # of tiles, threads/tile, # of sms, max # threads / sm
unsigned int get_num_rounds (unsigned int num_tiles, unsigned int threads_tile, unsigned int num_sms, unsigned int max_sm);

// given a vector length, return # cycles to compute vector operation. Can include: addition/multiplication/reduction
unsigned int vector_op(unsigned int len, unsigned int warp_size);

// given tile dimensions and the # of elements each thread must process in the tile, 
// return # cycles to compute the tile associated with those parameters
// this method assumes all data is already present in registers
unsigned int tile_op_1(unsigned int len, unsigned int ht, unsigned int elems_thread, unsigned int warp_size);

// given tile dimensions and the # of elements each thread must process in the tile, 
// return # cycles to compute the tile associated with those parameters
// this method assumes all data is already present in scratchpad memory (but not registers)
unsigned int tile_op_2(unsigned int len, unsigned int ht, unsigned int elems_thread, unsigned int warp_size);

// given # of cycles and a clock rate (MHz)
// return # of us to execute those cycles
unsigned int cycles_to_time(unsigned int cycles, unsigned int clock);

#endif
