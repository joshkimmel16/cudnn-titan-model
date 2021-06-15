# TitanV Model for CUDNN

This application is meant to model the performance of the [NVIDIA TitanV GPU](https://www.nvidia.com/en-us/titan/titan-v/) executing the classifier deep learning kernel implemented [CUDNN](https://developer.nvidia.com/cudnn). The goal is, for arbitrary parameters, to return the execution time as it would be observed running the CUDNN kernel on a TitanV GPU.

## Source Code Layout

The following is a high-level overview of the C++ repository:

* __include__ => Header files.
* __src__ => Implementation files
* __run_benchmarks.sh__ => For testing the model under various parameters.
* __cudnn_experiments_output__ => Results of CUDNN under various parameters.
* __model_experiments_output__ => Results of the model under various parameters.

## Prerequisites

* [cmake](https://cmake.org/install/) installed
* Some C/C++ compiler that is compatible with cmake.

## Building the Model

To build the model, use the following commands:

```
mkdir -p build
cd build
cmake ..
make
```

## Running the Model

To run the Python script (after it is properly configured), use the following command:

```
build/bin/model [-ni [input_dimension]] [-nn [output_dimension]] [-ti [input_tile_size]] [-tn [output_tile_size]] [-ty [type]]
```

* __ni__ => (Optional, unsigned int > 0) The size of the input vector. Default = 1024.
* __nn__ => (Optional, unsigned int > 0) The size of the output vector. Default = 1024.
* __ti__ => (Optional, unsigned int > 0) The tile size along the input dimension. Default = 32.
* __tn__ => (Optional, unsigned int > 0) The tile size along the output dimension. Default = 32.
* __ty__ => (Optional, Enum) The type of analysis to perform. Options are: REGISTER=0, SCRATCHPAD=1, L2=2, MEMORY=3 (default).

The type imposes various assumptions on the analysis that could be useful for determining bottlenecks and/or architectural properties of the system. REGISTER assumes that all data is pre-loaded into registers. SCRATCHPAD assumes that all data is present in fast scratchpad memory. L2 assumes all data is present in the L2 cache. MEMORY models the true execution as it would take place in the GPU.

## Authors

* **Josh Kimmel**
* **Hannah Nguyen**
* **Daniel Ahn**
* **Adrien Hadj-Chaib**

