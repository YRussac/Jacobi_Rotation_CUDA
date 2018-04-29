# export PATH=/usr/local/cuda-8.0/bin:$PATH

nvcc main.cu -o main -std=c++11

./main