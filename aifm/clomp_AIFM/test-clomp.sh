#!/bin/bash

source ./../shared.sh

function run_AIFM_clomp {    
    echo "Running $1 $2 $3 $4 $5 $6 $7 $8"
    sudo stdbuf -o0 sh -c "$1 $AIFM_PATH/configs/client.config \
                           $MEM_SERVER_DPDK_IP:$MEM_SERVER_PORT $2 $3 $4 $5 $6 $7 $8"
}

function run_clomp {
    rerun_local_iokerneld
    rerun_mem_server
    run_AIFM_clomp ./clomp 16 1 16 400 32 1 100
}

function cleanup {
    kill_local_iokerneld
    kill_mem_server
}

run_clomp
cleanup

exit 0
