#include "titanv.h"
#include "helpers.h"
#include <cerrno>
#include <thread>
#include <iostream>
#include <string>
#include <string.h>
#include <exception>

enum Type { REGISTER=0, SCRATCHPAD=1, L2=2, MEMORY=3 };

// helper method to call the appropriate analysis methods based on type
unsigned int tile_op (Type ty, unsigned int len, unsigned int ht, unsigned elems_thread, TitanV m) {
    if (ty == REGISTER) {
        return tile_op_1(len, ht, elems_thread, m);
    }
    else if (ty == SCRATCHPAD) {
        return tile_op_2(len, ht, elems_thread, m);
    }
    else if (ty == L2) { // TODO: implement tile_op_3
        return tile_op_1(len, ht, elems_thread, m);
    }
    else { // MEMORY (TODO: implement tile_op_4)
        return tile_op_1(len, ht, elems_thread, m);
    }
}

unsigned int n_i = 1024; // dimension of input vector (default = 1024)
unsigned int n_n = 1024; // dimension of output vector (default = 1024)
unsigned int t_i = 32; // tile width (default = 32)
unsigned int t_n = 32; // tile height (default = 32)
Type t = REGISTER; // which type of analysis to run (default=REGISTER)

int main(int argc, char* argv[]) 
{
    try
    {
        for (unsigned int i=1; i<argc; i+=2) { // loop through passed arguments
            if (i >= argc-1) { break; } // corner case where bad # of arguments passed
            if (strcmp(argv[i], "-ni") == 0) // input dimension
            {
                n_i = std::stoi(argv[i+1]);
            }
            else if (strcmp(argv[i], "-nn") == 0) // output dimension
            {
                n_n = std::stoi(argv[i+1]);
            }
            else if (strcmp(argv[i], "-ti") == 0) // tile width
            {
                t_i = std::stoi(argv[i+1]);
            }
            else if (strcmp(argv[i], "-tn") == 0) // tile height
            {
                t_n = std::stoi(argv[i+1]);
            }
            else if (strcmp(argv[i], "-ty") == 0) // type
            {
                t = (Type)std::stoi(argv[i+1]);
            }
        }

        TitanV machine = TitanV(); // instantiate TitanV

        // start by computing the number of tiles, threads per tile, and elements processed per thread
        // TODO: adjust this based on how CUDNN does it
        unsigned int tile_count = number_tiles(n_i, n_n, t_i, t_n);
        unsigned int threads_tile = threads_per_tile(t_i, t_n);
        unsigned int elements_per_thread = 1; // we assume that each tile is computed in 1 thread block
        if (threads_tile > machine.max_threads_block) { // # threads is too many for 1 thread block
            // have each thread do x times as much
            // this means we can reduce the # of threads
            // continue as such until we are below the max threads threshold
            unsigned int tmp_threads = threads_tile / 2;
            unsigned int out = 2;
            while (tmp_threads > machine.max_threads_block) {
                out++;
                tmp_threads = threads_tile / out;
            }
            threads_tile = tmp_threads;
            elements_per_thread = out;
        }

        std::cout << "Tile count: " << tile_count << std::endl;
        std::cout << "Threads per tile: " << threads_tile << std::endl;
        std::cout << "Elements per thread: " << elements_per_thread << std::endl;

        unsigned int tiles_per_sm = machine.max_threads_sm / threads_tile; // how many tiles can be mapped to 1 SM (want to round down in this case)
        unsigned int num_rounds = get_num_rounds(tile_count, threads_tile, machine); // how many sequential rounds of processing are necessary
        // TODO: at this point, make adjustments for memory limitations?

        std::cout << "Tiles per SM: " << tiles_per_sm << std::endl;
        std::cout << "Number of rounds: " << num_rounds << std::endl;

        // for each parallel round of processing, compute x tiles
        // where x is the # of tiles mapped to 1 SM
        unsigned int cycle_count = 0;
        unsigned int tiles_left = tile_count;
        for (unsigned i=0; i<num_rounds; i++) {
            unsigned int multiplier = (tiles_left < tiles_per_sm) ? 1 : tiles_per_sm; // special check to ensure that we're actually using the tile capacity...
            cycle_count += (multiplier * tile_op(t, t_i, t_n, elements_per_thread, machine));
            tiles_left -= (tiles_per_sm * machine.num_sms);
        }

        std::cout << "Cycles: " << cycle_count << std::endl;

        // compute # of us to run the computation
        unsigned int time = cycles_to_time(cycle_count, machine);

        std::cout << "Time: " << time << std::endl;

    }
    catch (std::exception& e)
    {
        std::cerr << "Error occurred! " << e.what();
        std::cerr << errno;
        std::cerr << strerror(errno);
        return 1;
    }

    return 0;
}