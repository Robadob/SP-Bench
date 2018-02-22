#!/bin/bash
nvcc main.cpp -I ../include/ -I ../common/ -I . -std=c++14 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_60,code=sm_60 -rdc=true -m 64 -g -G -D _DEBUG -o ../bin/x64/debug-bench
