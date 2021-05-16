#!/bin/bash

echo "----------------------------------------------"
echo "tile experiments:"
echo "----------------------------------------------"

TILE_HEIGHTS=(8 12 16 24 32 64)
TILE_WIDTHS=(8 12 16 24 32 64)

for height in "${TILE_HEIGHTS[@]}"
do
  for width in "${TILE_WIDTHS[@]}"
  do
    if [ "$height" -gt "$width" ]
    then
      continue
    fi
    echo "height is $height, width is $width"
    ./build/bin/model -tn $height -ti $width
    echo
  done
done

echo "----------------------------------------------"
echo "input/output vector size experiments"
echo "----------------------------------------------"

NI=(4096 8192 16384 25088 65536 131072)
NN=(512 1024 2048 4096)

for input_size in "${NI[@]}"
do
  for output_size in "${NN[@]}"
  do
    if [ "$output_size" -ge "$input_size" ]
    then
      continue
    fi
    # echo "std::make_tuple($input_size, 1, $output_size, false, false),"
    echo "input_size is $input_size, output_size is $output_size"
    ./build/bin/model -ni $input_size -nn $output_size
    echo
  done
done

## For CUDNN
## $ vim DeepBench/code/kernels/gemm_problems.h ==> code below
## $ cd DeepBench/code/nvidia
## $ make gemm
## $ bin/gemm_bench inference float

# std::vector<std::tuple<int, int, int, bool, bool>> training_set = {};

# std::vector<std::tuple<int, int, int, bool, bool>> inference_server_set = {
# std::make_tuple(4096, 1, 512, false, false),
# std::make_tuple(4096, 1, 1024, false, false),
# std::make_tuple(4096, 1, 2048, false, false),
# std::make_tuple(8192, 1, 512, false, false),
# std::make_tuple(8192, 1, 1024, false, false),
# std::make_tuple(8192, 1, 2048, false, false),
# std::make_tuple(8192, 1, 4096, false, false),
# std::make_tuple(16384, 1, 512, false, false),
# std::make_tuple(16384, 1, 1024, false, false),
# std::make_tuple(16384, 1, 2048, false, false),
# std::make_tuple(16384, 1, 4096, false, false),
# std::make_tuple(25088, 1, 512, false, false),
# std::make_tuple(25088, 1, 1024, false, false),
# std::make_tuple(25088, 1, 2048, false, false),
# std::make_tuple(25088, 1, 4096, false, false),
# std::make_tuple(65536, 1, 512, false, false),
# std::make_tuple(65536, 1, 1024, false, false),
# std::make_tuple(65536, 1, 2048, false, false),
# std::make_tuple(65536, 1, 4096, false, false),
# std::make_tuple(131072, 1, 512, false, false),
# std::make_tuple(131072, 1, 1024, false, false),
# std::make_tuple(131072, 1, 2048, false, false),
# std::make_tuple(131072, 1, 4096, false, false),
# };

# std::vector<std::tuple<int, int, int, bool, bool>> inference_device_set = {
# std::make_tuple(4096, 1, 512, false, false),
# std::make_tuple(4096, 1, 1024, false, false),
# std::make_tuple(4096, 1, 2048, false, false),
# std::make_tuple(8192, 1, 512, false, false),
# std::make_tuple(8192, 1, 1024, false, false),
# std::make_tuple(8192, 1, 2048, false, false),
# std::make_tuple(8192, 1, 4096, false, false),
# std::make_tuple(16384, 1, 512, false, false),
# std::make_tuple(16384, 1, 1024, false, false),
# std::make_tuple(16384, 1, 2048, false, false),
# std::make_tuple(16384, 1, 4096, false, false),
# std::make_tuple(25088, 1, 512, false, false),
# std::make_tuple(25088, 1, 1024, false, false),
# std::make_tuple(25088, 1, 2048, false, false),
# std::make_tuple(25088, 1, 4096, false, false),
# std::make_tuple(65536, 1, 512, false, false),
# std::make_tuple(65536, 1, 1024, false, false),
# std::make_tuple(65536, 1, 2048, false, false),
# std::make_tuple(65536, 1, 4096, false, false),
# std::make_tuple(131072, 1, 512, false, false),
# std::make_tuple(131072, 1, 1024, false, false),
# std::make_tuple(131072, 1, 2048, false, false),
# std::make_tuple(131072, 1, 4096, false, false),
# };
