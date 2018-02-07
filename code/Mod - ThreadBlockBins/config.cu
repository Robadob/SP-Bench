#define MOD_NAME "ThreadBlock Bin Scheduling"

//Options
#define THREADBLOCK_BINS

//Build
#include "BuildAll.cuh"

//Notes
//This strategy attempts to assign the processing fo each environmental bin to a seperate threadblock.
//This follows the example of existing frameworks such as fluids3 and LAMMPS (their code is less clear, but has similarities)
//Links collected Mid-January 2018
//https://github.com/rchoetzlein/fluids3/blob/95ebd2c22794cdf8531c3490b3f1e3f1bb505bc3/fluids/fluid_system_kern.cu#L320
//https://github.com/lammps/lammps/blob/fa4c7fc66462f61bbde34b4c8bd9604157c37b20/lib/gpu/lal_neighbor_gpu.cu#L141