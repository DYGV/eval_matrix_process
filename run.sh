#!/bin/sh
gcc ./serial_omp_vec_time.c -o serial_omp_vec_time -mavx2 -fopenmp
trap "exit 1" 2
for i in {1..10}
do
  echo "${i} --------------------------------------------------------"
  perf stat -e instructions,cache-references,cache-misses ./serial_omp_vec_time &> /dev/null
  if [ $? -gt 0 ]; then
    ./serial_omp_vec_time
  fi
done
