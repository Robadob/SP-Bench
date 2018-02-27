#!/bin/bash
nvcc main2.cpp ParamFactory.cpp ParamSet.cpp -I ../include/ -I ../common/ -I . -std=c++11 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_60,code=sm_60 -rdc=true -m 64 -o ../bin/x64/release-bench
