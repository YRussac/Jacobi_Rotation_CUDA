export PATH=/usr/local/cuda-8.0/bin:$PATH

nvcc src/main.cu -o build/main -std=c++11

./build/main

rm ./build/main