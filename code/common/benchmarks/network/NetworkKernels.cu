#include "NetworkKernels.cuh"
#include "NetworkModel.cuh"


//__device__ __constant__ unsigned int EDGE_COUNT;
//__device__ __constant__ unsigned int d_edgesPerVert;
//__device__ __constant__ unsigned int *d_vertexEdges;
//__device__ __constant__ float *d_edgeLen;
//__device__ __constant__ unsigned int *d_edgeCapacity;
//__device__ __constant__ DIMENSIONS_VEC *d_vertexLocs;
__global__ void
__launch_bounds__(64)
step_network_model(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut, const VertexData *vIn, VertexData *vOut)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= d_locationMessageCount)
        return;
    //Get my local values
#ifdef AOS_MESSAGES
    unsigned int myEdge = locationMessagesIn->location[id].x;
#else
    unsigned int myEdge = locationMessagesIn->locationX[id];
#endif
    const float myEdgeLen = d_edgeLen[myEdge];
    float distanceAlongEdge = vIn[id].edgeDistance;
    const unsigned int myVert = myEdge/3;
    //Move along edge
    distanceAlongEdge += vIn[id].speed;
    //If we have exceeded the end of the edge
    if (distanceAlongEdge >= myEdgeLen)
    {
        //Select a new edge
        const unsigned int newEdgeRng = (clock() + id + vIn[id].agentId) % d_edgesPerVert;//Pseudo rng [0-d_edgesPerVert)
        const unsigned int newEdge = (myVert*d_edgesPerVert) + newEdgeRng;
        assert(newEdge < EDGE_COUNT);
        const unsigned int myNewEdgeCapacity = d_edgeCapacity[newEdge];
        //Count the number of agents currently on the edge
        int ct = 0;
#if !defined(SHARED_BINSTATE)
        LocationMessage _lm;
        LocationMessage *lm = &_lm;
        locationMessagesIn->getFirstMessage(newEdge, &_lm);
#else
        LocationMessage *lm = locationMessagesIn->getFirstNeighbour(myLoc);
#endif
        while (lm)
        {
            ct++;
#if defined(SHARED_BINSTATE)
            lm = locationMessagesIn->getNextNeighbour(lm);//Returns a pointer to shared memory or 0
#else
            locationMessagesIn->getNextNeighbour(lm);
#endif
        }
        //Edge has space, switch
        if (ct<myNewEdgeCapacity)
        {
            myEdge = newEdge;
            //Reset distance along edge
            distanceAlongEdge = 0.0f;
        }
    }
    //Export changes
    vOut[id].edgeDistance = distanceAlongEdge;
#ifdef AOS_MESSAGES
    locationMessagesOut->location[id].x = myEdge;
#else
    locationMessagesOut->locationX[id] = myEdge;
#endif
}


__global__ void init_network(curandState *state, LocationMessages *locationMessages, VertexData *v)
{
    //This is copied frm init_particles, but i dont think we actually plan to use the locations.
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= d_locationMessageCount)
        return;
    //curand_unform returns 0<x<=1.0, not much can really do about 0 exclusive
    //negate and  + 1.0, to make  0<=x<1.0
#ifdef AOS_MESSAGES
    locationMessages->location[id].x = floor((-curand_uniform(&state[id]) + 1.0f) * EDGE_COUNT);
    locationMessages->location[id].y = 0;
#ifdef _3D
    locationMessages->location[id].z = 0;
#endif
#else
    locationMessages->locationX[id] = floor((-curand_uniform(&state[id]) + 1.0f) * EDGE_COUNT);
    locationMessages->locationY[id] = 0;
#ifdef _3D
    locationMessages->locationZ[id] = 0;
#endif
#endif
    //Network specific stuff
    //Edge which agent is currently on
    v->agentId = id;
    v->edgeDistance = 0;
    v->speed = 0.005 + (curand_uniform(&state[id]) / 20);
}
__global__ void init_network_uniform(LocationMessages *locationMessages, VertexData *v)
{
    //This is copied frm init_particles, but i dont think we actually plan to use the locations.
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= d_locationMessageCount)
        return;
    //curand_unform returns 0<x<=1.0, not much can really do about 0 exclusive
    //negate and  + 1.0, to make  0<=x<1.0
#ifdef AOS_MESSAGES
    locationMessages->location[id].x = id %EDGE_COUNT;
    locationMessages->location[id].y = 0;
#ifdef _3D
    locationMessages->location[id].z = 0;
#endif
#else
    locationMessages->locationX[id] = id %EDGE_COUNT;
    locationMessages->locationY[id] = 0;
#ifdef _3D
    locationMessages->locationZ[id] = 0;
#endif
#endif
    //Network specific stuff
    //Edge which agent is currently on
    v->agentId = id;
    v->edgeDistance = 0;
    v->speed = 0.01;
}