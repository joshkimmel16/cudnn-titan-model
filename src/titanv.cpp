#include "titanv.h"

TitanV::TitanV () {} // nothing to do in constructor

/*
unsigned int TitanV::load_registers (unsigned int vals) {
    // assume all register loads are coming from scratchpad
    // assume this takes a single cycle per load of 1 warp's worth of values
    unsigned int num_atomic = 1;
    return num_atomic * ((vals % warp_size == 0) ? vals/warp_size : (vals/warp_size) + 1);
}

unsigned int TitanV::load_scratchpad (unsigned int vals) {
    // assume all scratchpad loads are coming from L2 cache
    // # cycles = # MBs * # seconds / MB * # cycles / second
    return (vals*val_size) * gpu_clock / l2_bw / (1024*1024);
}

unsigned int TitanV::load_l2 (unsigned int vals) {
    // assume all L2 loads are coming from main memory
    // # cycles = # MBs * # seconds / MB * # cycles / second
    return (vals*val_size) * gpu_clock / global_mem_bw / (1024*1024);
}

unsigned int TitanV::store_scratchpad (unsigned int vals) {
    // assume all scratchpad stores are coming from registers
    // assume this takes a single cycle per store of 1 warp's worth of values
    unsigned int num_atomic = 1;
    return num_atomic * ((vals % warp_size == 0) ? vals/warp_size : (vals/warp_size) + 1);
}

unsigned int TitanV::store_l2 (unsigned int vals) {
    // assume all L2 stores are coming from scratchpad
    // # cycles = # MBs * # seconds / MB * # cycles / second
    return (vals*val_size) * gpu_clock / l2_bw / (1024*1024);
}

unsigned int TitanV::store_mem (unsigned int vals) {
    // assume all memory stores are coming from L2
    // # cycles = # MBs * # seconds / MB * # cycles / second
    return (vals*val_size) * gpu_clock / global_mem_bw / (1024*1024);
}
*/