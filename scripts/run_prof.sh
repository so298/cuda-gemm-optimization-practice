#!/bin/bash

set -eux

# if no argument is passed, use 1024 as default
size=${1:-2048}

for src_file in $(ls ./src/*.cu); do
    executable=$(basename $src_file .cu)
    ncu -s 50 -c 1 --csv -f -o prof/${executable} ./exe/${executable} -n $size -k $size -m $size -i 100
    ncu --import prof/${executable}.ncu-rep --csv --log-file prof/${executable}.csv
done
