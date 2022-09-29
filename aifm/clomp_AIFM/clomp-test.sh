#!/bin/bash

source ./../shared.sh


function run_clomp {
    echo "Running ./clomp 16 1 16 400 32 1 100"
    rerun_local_iokerneld
    rerun_mem_server
    run_program ./clomp
}

function cleanup {
    kill_local_iokerneld
    kill_mem_server
}

run_clomp
cleanup

exit 0
