#!/bin/bash

source clomp-shared.sh


function run_single_test {
    echo "Running test $1..."
    rerun_local_iokerneld
    if [[ $1 == *"tcp"* ]]; then
        rerun_mem_server
    fi
    run_program ./bin/$1
    
}

function cleanup {
    kill_local_iokerneld
    kill_mem_server
}

test=test_tcp_pointer_swap

run_single_test $test
cleanup

exit 0
