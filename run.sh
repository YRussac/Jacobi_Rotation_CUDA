export PATH=/usr/local/cuda-9.1/bin:$PATH

nvcc src/main.cu -o build/main -std=c++11 -w

./build/main

rm ./build/main
