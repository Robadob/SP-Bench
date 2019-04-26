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
step_network_model_old(LocationMessages *locationMessagesIn, LocationMessages *locationMessagesOut, const VertexData *vIn, VertexData *vOut)
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
    const unsigned int myDestinationVert = d_vertexEdges[myEdge];
    //Move along edge
    distanceAlongEdge += vIn[id].speed;
    //If we have exceeded the end of the edge
    if (distanceAlongEdge >= myEdgeLen)
    {
        //Select a new edge
        const unsigned int newEdgeRng = (clock() + id + vIn[id].agentId) % d_edgesPerVert;//Pseudo rng [0-d_edgesPerVert)
        //const unsigned int newEdge = (myVert*d_edgesPerVert) + newEdgeRng;
        
        unsigned int bestEdge = UINT_MAX;
        unsigned int lowestCost = UINT_MAX;
        //Iterate all connected edges
        for (unsigned int i = 0; i < d_edgesPerVert;++i)
        {
            //Shuffle which edge each agent accesses first
            const unsigned int newEdge = (myDestinationVert*d_edgesPerVert) + ((i + newEdgeRng) % d_edgesPerVert);
            int ct = 0;
#if !defined(SHARED_BINSTATE)
            LocationMessage _lm;
            LocationMessage *lm = &_lm;
            locationMessagesIn->getFirstMessage(newEdge, &_lm);
#else
            LocationMessage *lm = locationMessagesIn->getFirstNeighbour(myLoc);
#endif
            if (lm)
            {
                do
                {
                    ct++;
#if defined(SHARED_BINSTATE)
                    lm = locationMessagesIn->getNextNeighbour(lm);//Returns a pointer to shared memory or 0
                } while (lm)
#else
                } while (locationMessagesIn->getNextNeighbour(lm));
#endif  
                if (ct<lowestCost)
                {
                    lowestCost = ct;
                    bestEdge = newEdge;
                }
            }
        }
        assert(bestEdge < EDGE_COUNT);
        //We could get rid of the capacity element
        const unsigned int myNewEdgeCapacity = d_edgeCapacity[bestEdge];
        //Edge has space, switch
        if (lowestCost<myNewEdgeCapacity)
        {
            myEdge = bestEdge;
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


//uint myEdge;
//float myEdgeProgress, mySpeed;
//uint myVert = myEdge / EDGES_PER_VERT;
//myEdgeProgress += mySpeed;
//uint nextVert = edges[myEdge].destination;
//uint nextEdge = nextVert * EDGES_PER_VERT;
//uint minId = UINT_MAX, minCount = UINT_MAX;
//for (i = 0; i<EDGES_PER_VERT; i++)
//{
//    uint edge = nextEdge + i;
//    uint count = 0;
//    foreach message in bin[edge]
//    {
//        count++;
//    }
//        if (count - edges[edge].capacity<minCount)
//        {
//        minCount = count - edges[edge].capacity;
//        minId = edge;
//        }
//}
//if (myEdgeProgress >= edges[myEdge].length)
//{
//    if (minCount<UINT_MAX && minCount>0)
//    {
//        myEdge = minId;
//        myEdgeProgress = 0;
//    }
//}
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
    float myEdgeProgress = vIn[id].edgeDistance + vIn[id].speed;
    if (myEdgeProgress >= myEdgeLen)
    {    
        //Decide next edge
        const unsigned int nextVert = d_vertexEdges[myEdge];
        const unsigned int nextEdge = nextVert * d_edgesPerVert;
        unsigned int minId = UINT_MAX, maxSpace = 0;
        const unsigned int newEdgeRng = (clock() + id + vIn[id].agentId) % d_edgesPerVert;//Pseudo rng [0-d_edgesPerVert)
        for (unsigned int i = 0; i<d_edgesPerVert; i++)
        {
            const unsigned int edge = nextEdge + ((newEdgeRng + i) % d_edgesPerVert);
            unsigned int count = 0;
            //foreach message in bin[edge]
#if !defined(SHARED_BINSTATE)
            LocationMessage _lm;
            LocationMessage *lm = &_lm;
            locationMessagesIn->getFirstMessage(edge, &_lm);
#else
            LocationMessage *lm = locationMessagesIn->getFirstNeighbour(myLoc);
#endif
            if (lm)
            {
                do
                {
                    count++;
#if defined(SHARED_BINSTATE)
                    lm = locationMessagesIn->getNextNeighbour(lm);//Returns a pointer to shared memory or 0
                } while (lm)
#else
            } while (locationMessagesIn->getNextNeighbour(lm));
#endif  
                if (count<d_edgeCapacity[edge] && d_edgeCapacity[edge] - count>maxSpace)
                {
                    maxSpace = d_edgeCapacity[edge] - count;
                    minId = edge;
                }
            }
        }
        if (maxSpace>0)
        {
            myEdge = minId;
            myEdgeProgress = 0;
        }
    }
    locationMessagesOut->locationX[id] = myEdge;
    vOut[id].edgeDistance = myEdgeProgress;
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
    v->speed = 0.75 + (curand_uniform(&state[id]) / 20);
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
    v->speed = 0.75;
}