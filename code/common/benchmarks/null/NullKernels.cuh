#ifndef __NullKernels_cuh__
#define __NullKernels_cuh__

#include "near_neighbours/Neighbourhood.cuh"

__global__ void step_null_model(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut, DIMENSIONS_VEC *result);

#endif //__NullKernels_cuh__
