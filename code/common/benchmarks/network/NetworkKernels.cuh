#ifndef __NetworkKernels_cuh__
#define __NetworkKernels_cuh__

#include "near_neighbours/Neighbourhood.cuh"//Remove with templating if possible

__global__ void step_network_model(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut);
#endif //__NetworkKernels_cuh__
