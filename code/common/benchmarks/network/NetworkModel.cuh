#ifndef __NetworkModel_cuh__
#define __NetworkModel_cuh__

#include <math.h>
#include "results.h"
#include "../core/CoreModel.cuh"
#include "NetworkKernels.cuh"
struct VertexData
{
    unsigned int agentId;
    float edgeDistance;
    float speed;
};
class NetworkModel : public CoreModel
{
public:
    NetworkModel(
        const unsigned int agents = 16384,
        const unsigned int vertices = 5000,
        const unsigned int edgesPerVertex = 3,
        const unsigned int capacityModifier = 2//Value of 0 will set capacity to edge as minimum that will fit max agents, +1 increments capacity per edge by 1 etc
        );
    //Returns the time taken
    const Time_Step step() override;
    //Network benchmark requires custom init, as agents are not just spawned within a 2D/3D cube
    const Time_Init initPopulation(const unsigned long long rngSeed = 12) override;
    const Time_Init initPopulationUniform() override { return initPopulation(0); }
    const Time_Init initPopulationClusters(const unsigned int clusterCount, const float clusterRad, const unsigned int agentsPerCluster, const unsigned long long rngSeed = 12) 
    { fprintf(stderr, "NetworkModel does not support Cluster init, falling back to uniform init.\n"); return initPopulation(0); }
private:
    void launchStep();//Launches step_model kernel
    std::shared_ptr<SpatialPartitionExt<VertexData>> spatialPartition;
    //If values are constant, might aswell make them public, save writing accessor methods
public:
    std::shared_ptr<SpatialPartition> getPartition() override { return spatialPartition; }
    std::shared_ptr<SpatialPartitionExt<VertexData>> getPartitionExt() { return spatialPartition; }
    const unsigned int vCount;
    const unsigned int edgesPer;
    const unsigned int CAPACITY_MOD;
    unsigned int *hd_vertexEdges;
    float *hd_edgeLen;
    unsigned int *hd_edgeCapacity;
    DIMENSIONS_VEC *hd_vertexLocs;
};

//Required to remove extern for RDC=false
__device__ __constant__ unsigned int EDGE_COUNT;
__device__ __constant__ unsigned int d_edgesPerVert;
__device__ __constant__ unsigned int *d_vertexEdges;
__device__ __constant__ float *d_edgeLen;
__device__ __constant__ unsigned int *d_edgeCapacity;
__device__ __constant__ DIMENSIONS_VEC *d_vertexLocs;

NetworkModel::NetworkModel(
    const unsigned int agents,
    const unsigned int vertices,
    const unsigned int edgesPerVertex,
    const unsigned int capacityModifier
    )
    : CoreModel(agents)
    , spatialPartition(std::make_shared<SpatialPartitionExt<VertexData>>(vertices, agents))
    , vCount(vertices)
    , edgesPer(edgesPerVertex)
    , CAPACITY_MOD(capacityModifier)
{
    CUDA_CALL(cudaMemcpyToSymbol(d_edgesPerVert, &edgesPer, sizeof(unsigned int)));
    unsigned int edgeMax = vertices * edgesPerVertex;
    CUDA_CALL(cudaMemcpyToSymbol(EDGE_COUNT, &edgeMax, sizeof(unsigned int)));
}

const Time_Init NetworkModel::initPopulation(const unsigned long long rngSeed)
{

    cudaEvent_t start_overall, start_curand, start_kernel, start_pbm, start_free, stop_overall;
    cudaEventCreate(&start_overall);
    cudaEventCreate(&start_curand);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&start_pbm);
    cudaEventCreate(&start_free);
    cudaEventCreate(&stop_overall);

    //Start overall timer
    cudaEventRecord(start_overall);

    //Generate network
    cudaMalloc(&hd_vertexEdges, sizeof(unsigned int)*edgesPer*vCount);
    CUDA_CALL(cudaMemcpyToSymbol(d_vertexEdges, &hd_vertexEdges, sizeof(unsigned int *)));
    cudaMalloc(&hd_edgeLen, sizeof(float)*edgesPer*vCount);
    CUDA_CALL(cudaMemcpyToSymbol(d_edgeLen, &hd_edgeLen, sizeof(float *)));
    cudaMalloc(&hd_edgeCapacity, sizeof(unsigned int)*edgesPer*vCount);
    CUDA_CALL(cudaMemcpyToSymbol(d_edgeCapacity, &hd_edgeCapacity, sizeof(unsigned int *)));
    cudaMalloc(&hd_vertexLocs, sizeof(DIMENSIONS_VEC)*vCount);
    CUDA_CALL(cudaMemcpyToSymbol(d_vertexLocs, &hd_vertexLocs, sizeof(DIMENSIONS_VEC *)));
    {
        //Alloc temp host copies of arrays
        unsigned int *h_vertexEdges = (unsigned int *)malloc(sizeof(unsigned int)*edgesPer*vCount);
        float *h_edgeLen = (float *)malloc(sizeof(float)*edgesPer*vCount);
        unsigned int *h_edgeCapacity = (unsigned int *)malloc(sizeof(unsigned int)*edgesPer*vCount);
        DIMENSIONS_VEC *h_vertexLocs = (DIMENSIONS_VEC *)malloc(sizeof(DIMENSIONS_VEC)*vCount);

        //Fill arrays
        {
            std::mt19937 gen(12);//Fixed seed
            std::mt19937 gen2(21);//Fixed seed (use seperate seed, so we don't interfer with other thing if we strip out location in future)
            std::uniform_real_distribution<float> normal(0.0f, 1.0f);
            const unsigned int edgeCapacity = (unsigned int)ceil(CAPACITY_MOD + (agentMax / (vCount*edgesPer)));
            for (unsigned int i = 0; i < vCount;++i)
            {//Iterate vertices
                for (unsigned int j = 0; j < edgesPer;++j)
                {//Iterate edges per vertex
                    const unsigned int edgeOffset = (i*edgesPer) + j;
                    h_vertexEdges[edgeOffset] = (i + j*(int)(vCount / (edgesPer + 1.7))) % vCount; //Basic formula to spread edges in some kind of pattern
                    if (h_vertexEdges[edgeOffset] == i) //Vertex shoouldn't link to itself
                        h_vertexEdges[edgeOffset]++;
                    if (h_vertexEdges[edgeOffset] >= vCount) //Links to a valid vertex
                        h_vertexEdges[edgeOffset] = 0;
                    h_edgeLen[edgeOffset] = 1.0f + normal(gen); //Lengths vary from 1-2
                    h_edgeCapacity[edgeOffset] = edgeCapacity;
                }
#if DIMENSIONS==3
                h_vertexLocs[i] = glm::vec3(normal(gen2), normal(gen2), normal(gen2));
#elif DIMENSIONS==2
                h_vertexLocs[i] = glm::vec2(normal(gen2), normal(gen2));
#else
#error Invalid DIMENSIONS value, only 2 and 3 are suitable
#endif
            }
        }

        //Copy to device
        CUDA_CALL(cudaMemcpy(hd_vertexEdges, h_vertexEdges, sizeof(unsigned int)*edgesPer*vCount, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(hd_edgeLen, h_edgeLen, sizeof(float)*edgesPer*vCount, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(hd_edgeCapacity, h_edgeCapacity, sizeof(unsigned int)*edgesPer*vCount, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(hd_vertexLocs, h_vertexLocs, sizeof(DIMENSIONS_VEC)*vCount, cudaMemcpyHostToDevice));

        //Free temp copies
        free(h_vertexEdges);
        free(h_edgeLen);
        free(h_edgeCapacity);
        free(h_vertexLocs);
    }


    //start curand timer
    cudaEventRecord(start_curand);

    //Generate curand
    curandState *d_rng;
    CUDA_CALL(cudaMalloc(&d_rng, agentMax*sizeof(curandState)));
    //Arbitrary thread block sizes (speed not too important during one off initialisation)
    unsigned int initThreads = 512;
    unsigned int initBlocks = (agentMax / initThreads) + 1;
    getPartition()->setLocationCount(agentMax);
    if (rngSeed != 0)
    {
        init_curand <<<initBlocks, initThreads>>>(d_rng, agentMax, rngSeed);//Defined in CircleKernels.cuh
        CUDA_CALL(cudaDeviceSynchronize());
    }

    //End curand timer/start kernel timer
    cudaEventRecord(start_kernel);

    //Generate initial states, and store in location messages
    LocationMessages *d_lm = getPartitionExt()->d_getLocationMessages();
    VertexData *d_ext = getPartitionExt()->d_getExtMessages();
    if (rngSeed != 0)
    {
        init_network <<<initBlocks, initThreads>>>(d_rng, d_lm, d_ext);
    }
    else
    {
        init_network_uniform <<<initBlocks, initThreads>>>(d_lm, d_ext);
    }
    CUDA_CALL(cudaDeviceSynchronize());

    //End kernel timer/start pbm timer
    cudaEventRecord(start_pbm);

    //generate pbm
    getPartition()->buildPBM();//This may error due to attempt to deallocate unallocated textures

    //End pbm timer/start free timer
    cudaEventRecord(start_free);

    //Free curand
    CUDA_CALL(cudaFree(d_rng));

    //End overall timer
    cudaEventRecord(stop_overall);

    //Calculate return struct
    //cudaEventSynchronize(start_overall);//Only really required to synchronise the last event
    //cudaEventSynchronize(start_curand);
    //cudaEventSynchronize(start_kernel);
    //cudaEventSynchronize(start_pbm);
    //cudaEventSynchronize(start_free);
    cudaEventSynchronize(stop_overall);
    Time_Init rtn;

    cudaEventElapsedTime(&rtn.overall, start_overall, stop_overall);
    cudaEventElapsedTime(&rtn.initCurand, start_curand, start_kernel);
    cudaEventElapsedTime(&rtn.kernel, start_kernel, start_pbm);
    cudaEventElapsedTime(&rtn.pbm, start_pbm, start_free);
    cudaEventElapsedTime(&rtn.freeCurand, start_free, stop_overall);

    return rtn; 
}
const Time_Step NetworkModel::step()
{
    cudaEvent_t start_overall, start_texture, stop_overall;
    cudaEventCreate(&start_overall);
    cudaEventCreate(&start_texture);
    cudaEventCreate(&stop_overall);

    //Start overall timer
    cudaEventRecord(start_overall);

    //Run single iteration of model
    launchStep();

    //End kernel timer/start texture reset timer
    cudaEventRecord(start_texture);

    //Rebuild PBM
    PBM_Time p = spatialPartition->buildPBM();

    //End overall timer
    cudaEventRecord(stop_overall);

    //Calculate return struct
    //cudaEventSynchronize(start_overall);//Only really required to synchronise the last event
    //cudaEventSynchronize(start_texture);
    cudaEventSynchronize(stop_overall);
    Time_Step rtn;

    cudaEventElapsedTime(&rtn.overall, start_overall, stop_overall);
    cudaEventElapsedTime(&rtn.kernel, start_overall, start_texture);
    cudaEventElapsedTime(&rtn.texture, start_texture, stop_overall);
    rtn.pbm = p;

    return rtn;
}
int requiredSM_stepNetworkModel(int blockSize)
{
    return SpatialPartition::requiredSM(blockSize);
}
void NetworkModel::launchStep()
{
    int minGridSize, blockSize;   // The launch configurator returned block size 
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, step_circles_model, requiredSM_stepCirclesModel, 0);//random 128
    // Round up according to array size
    int gridSize = (agentMax + blockSize - 1) / blockSize;
    LocationMessages *d_lm = spatialPartition->d_getLocationMessages();
    LocationMessages *d_lm2 = spatialPartition->d_getLocationMessagesSwap();
    VertexData *d_ext = spatialPartition->d_getExtMessages();
    VertexData *d_ext2 = spatialPartition->d_getExtMessagesSwap();
    //Launch kernel
    step_network_model << <gridSize, blockSize, requiredSM_stepNetworkModel(blockSize) >> >(d_lm, d_lm2, d_ext, d_ext2);
    //Swap
    spatialPartition->swap();
    //Wait for return
    CUDA_CALL(cudaDeviceSynchronize());
}

#endif //__NetworkModel_cuh__
