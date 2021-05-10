#ifndef TITANV_H
#define TITANV_H

// references:
// https://www.techpowerup.com/gpu-specs/titan-v.c3051 => global memory bandwidth reported implies DDR (0.5 cycles/transfer)

struct TitanV {
    unsigned int global_mem_cap = 12037; // MB's
    unsigned int global_mem_bw = (3072/8) * 2 * 850; // bus width (bytes)/transfer / cycles/transfer (this value is an assumption!) * bus clock (M cycles/s) = MB/s
    
    unsigned int num_cores = 5120; // # cores
    unsigned int gpu_clock = 1455; // MHz
    unsigned int warp_size = 32; // # threads / warp
    unsigned int num_sms = 80; // # SM's in the system
    unsigned int max_threads_block = 1024; // max # threads per block
    unsigned int max_threads_sm = 2048; // max # threads per block
    
    unsigned int l2_cap = 4718592 / 1024; // MB's
    unsigned int l2_bw = 1000; // => MB/s TODO: what should this be??
    unsigned int constant_cap = 65536 / 1024; // MB's
    unsigned int shared_cap = 49152 / 1024; // MB's

    unsigned int val_size = 8; // bytes (FP64)
    unsigned int cpi = 1; // # of cycles (in equilibrim) per instruction
    double max_lat_hide = 0.8; // at most, 80% of latency can be hidden (TODO: refine this value) 

    TitanV(); // default constructor
    
    /*
    unsigned int load_registers(unsigned int vals); // given the number of data elements to load, return # cycles to load them from scratchpad into registers
    unsigned int load_scratchpad (unsigned int vals); // given the number of data elements to load, return # cycles to load them from L2 into scratchpad
    unsigned int load_l2 (unsigned int vals); // given the number of data elements to load, return # cycles to load them from memory into L2
    unsigned int store_scratchpad(unsigned int vals); // given the number of data elements to store, return # cycles to store them from registers into scratchpad
    unsigned int store_l2 (unsigned int vals); // given the number of data elements to store, return # cycles to store them from scratchpad into L2
    unsigned int store_mem (unsigned int vals); // given the number of data elements to store, return # cycles to store them from L2 into memory
    */
};

#endif