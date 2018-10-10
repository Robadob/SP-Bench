#ifndef __NetworkKernels_cuh__
#define __NetworkKernels_cuh__

//#pragma warning(disable:4244)
//#pragma warning(disable:4305)
#include <curand_kernel.h>
//#pragma warning (default : 4244)
//#pragma warning (default : 4305)

#include "near_neighbours/Neighbourhood.cuh"//Remove with templating if possible

__global__ void step_network_model(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut);
struct VertexData;
__global__ void init_network(curandState *state, LocationMessages *locationMessages, VertexData *v);
#endif //__NetworkKernels_cuh__
