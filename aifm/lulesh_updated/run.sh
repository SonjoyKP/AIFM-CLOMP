#!/bin/bash

source ../shared.sh

#cache_sizes_arr=( 40000 600 )

#sudo pkill -9 main
#for cache_size in ${cache_sizes_arr[@]}
#do
#    sed "s/constexpr static uint64_t kCacheSize.*/constexpr static uint64_t kCacheSize = $cache_size * Region::kSize;/g" main.cpp -i
make clean
make -j
rerun_local_iokerneld
rerun_mem_server
    #run_program ./lulesh2.0 1>log.$cache_size 2>&1
run_program ./lulesh2.0
#done
kill_local_iokerneld
kill_mem_server



