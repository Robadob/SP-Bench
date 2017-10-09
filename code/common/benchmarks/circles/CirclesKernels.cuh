#ifndef __CirclesKernels_cuh__
#define __CirclesKernels_cuh__

#include "near_neighbours/Neighbourhood.cuh"//Remove with templating if possible

__global__ void step_circles_model(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut);
#endif
