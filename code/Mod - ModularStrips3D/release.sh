#!/bin/bash
DIR_NAME_CLEAN=$(echo -e "${PWD##*/}" | tr -d '[:space:]')
nvcc config.cu -I ../include/ -I ../common/ -I . -std=c++11 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_60,code=sm_60 -rdc=true -m 64 -o ../bin/x64/release-${DIR_NAME_CLEAN,,}
