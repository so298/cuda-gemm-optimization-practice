#!/bin/bash

set -eux

# if no argument is passed, use 1024 as default
size=${1:-1024}

for executable in $(ls ./exe); do
    ./exe/${executable} -n $size -k $size -m $size -i 100
done