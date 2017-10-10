#ifndef __CoreModel_cuh__
#define __CoreModel_cuh__

#include "CoreKernels.cuh"
#include <random>

class CoreModel
{
public:
    CoreModel(const unsigned int agentMax)
        :agentMax(agentMax)
    { }
    virtual ~CoreModel(){};
    //Returns the time taken
    virtual const Time_Init initPopulation(const unsigned long long rngSeed = 12);
    virtual const Time_Init initPopulationUniform();
    virtual const Time_Init initPopulationClusters(const unsigned int clusterCount, const float clusterRad, const unsigned long long rngSeed = 12);
    virtual const Time_Step step()=0;
    virtual std::shared_ptr<SpatialPartition> getPartition() = 0;
    const unsigned int agentMax;
};

const Time_Init CoreModel::initPopulationUniform()
{
    return initPopulation(0);
}

const Time_Init CoreModel::initPopulation(const unsigned long long rngSeed)
{
    cudaEvent_t start_overall, start_kernel, start_pbm, start_free, stop_overall;
    cudaEventCreate(&start_overall);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&start_pbm);
    cudaEventCreate(&start_free);
    cudaEventCreate(&stop_overall);

    //Start overall timer
    cudaEventRecord(start_overall);

    //Generate curand
    curandState *d_rng;
    CUDA_CALL(cudaMalloc(&d_rng, agentMax*sizeof(curandState)));
    //Arbitrary thread block sizes (speed not too important during one off initialisation)
    unsigned int initThreads = 512;
    unsigned int initBlocks = (agentMax / initThreads) + 1;
    getPartition()->setLocationCount(agentMax);
    if (rngSeed != 0)
    {
        init_curand << <initBlocks, initThreads >> >(d_rng, rngSeed);//Defined in CircleKernels.cuh
        CUDA_CALL(cudaDeviceSynchronize());
    }

    //End curand timer/start kernel timer
    cudaEventRecord(start_kernel);

    //Generate initial states, and store in location messages
    LocationMessages *d_lm = getPartition()->d_getLocationMessages();
    if (rngSeed != 0)
    {
        init_particles << <initBlocks, initThreads >> >(d_rng, d_lm);
    }
    else
    {
#if DIMENSIONS==3
        int particlesPerDim = cbrt(agentMax)+1;
#elif DIMENSIONS==2
        int particlesPerDim = sqrt(agentMax)+1;
#else
#error Invalid DIMENSIONS value, only 2 and 3 are suitable
#endif
        DIMENSIONS_VEC offset = getPartition()->getEnvironmentDimensions() / (float)particlesPerDim;
        init_particles_uniform << <initBlocks, initThreads >> >(d_lm, particlesPerDim, offset);
        //Not sure why this ifdef is required
#ifdef _GL
        int bins = glm::compMul(getPartition()->getGridDim());
        int spareP = agentMax%bins;
        fprintf(stderr, "Bins: %i, Agents: %i\n", bins, agentMax);
        if (spareP>0)
        {
            fprintf(stderr, "Warning %.1f%% of bins have an extra agent!\n", (spareP / (float)bins) * 100);
        }
#endif
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
    //cudaEventSynchronize(start_kernel);
    //cudaEventSynchronize(start_pbm);
    //cudaEventSynchronize(start_free);
    cudaEventSynchronize(stop_overall);
    Time_Init rtn;

    cudaEventElapsedTime(&rtn.overall, start_overall, stop_overall);
    cudaEventElapsedTime(&rtn.initCurand, start_overall, start_kernel);
    cudaEventElapsedTime(&rtn.kernel, start_kernel, start_pbm);
    cudaEventElapsedTime(&rtn.pbm, start_pbm, start_free);
    cudaEventElapsedTime(&rtn.freeCurand, start_free, stop_overall);

    return rtn;
}

const Time_Init CoreModel::initPopulationClusters(const unsigned int clusterCount, const float clusterRad, const unsigned long long rngSeed)
{
    cudaEvent_t start_overall, start_kernel, start_pbm, start_free, stop_overall;
    cudaEventCreate(&start_overall);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&start_pbm);
    cudaEventCreate(&start_free);
    cudaEventCreate(&stop_overall);

    //Start overall timer
    cudaEventRecord(start_overall);

    //Generate curand
    curandState *d_rng;
    CUDA_CALL(cudaMalloc(&d_rng, agentMax*sizeof(curandState)));
    //Arbitrary thread block sizes (speed not too important during one off initialisation)
    unsigned int initThreads = 512;
    unsigned int initBlocks = (agentMax / initThreads) + 1;
    getPartition()->setLocationCount(agentMax);
    if (rngSeed != 0)
    {
        init_curand << <initBlocks, initThreads >> >(d_rng, rngSeed);//Defined in CircleKernels.cuh
        CUDA_CALL(cudaDeviceSynchronize());
    }

    //End curand timer/start kernel timer
    cudaEventRecord(start_kernel);

    //Generate initial states, and store in location messages
    LocationMessages *d_lm = getPartition()->d_getLocationMessages();
    const unsigned int agentsPerCluster = (agentMax / clusterCount) + 1;
    const glm::vec3 envMin = getPartition()->getEnvironmentMin() + glm::vec3(clusterRad);
    const glm::vec3 envMax = getPartition()->getEnvironmentMax() - glm::vec3(clusterRad);
    std::default_random_engine rng((unsigned int)rngSeed);
    std::uniform_real_distribution<float> rng_x(envMin.x, envMax.x);
    std::uniform_real_distribution<float> rng_y(envMin.y, envMax.y);
#ifdef _3D
    std::uniform_real_distribution<float> rng_z(envMin.z, envMax.z);
#endif
    initBlocks = (agentsPerCluster / initThreads) + 1;
    const float clusterWidth = 2 * clusterRad;
    for (unsigned int i = 0; i < clusterCount;++i)
    {
        //startIndex, limit, clusterCenter, clusterRad
        DIMENSIONS_VEC clusterCenter;
        clusterCenter.x = rng_x(rng);
        clusterCenter.y = rng_y(rng); 
#ifdef _3D
        clusterCenter.z = rng_z(rng);
#endif
        init_particles_clusters << <initBlocks, initThreads >> >(d_rng, d_lm, i*agentsPerCluster, agentsPerCluster, clusterCenter, clusterWidth);
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
    //cudaEventSynchronize(start_kernel);
    //cudaEventSynchronize(start_pbm);
    //cudaEventSynchronize(start_free);
    cudaEventSynchronize(stop_overall);
    Time_Init rtn;

    cudaEventElapsedTime(&rtn.overall, start_overall, stop_overall);
    cudaEventElapsedTime(&rtn.initCurand, start_overall, start_kernel);
    cudaEventElapsedTime(&rtn.kernel, start_kernel, start_pbm);
    cudaEventElapsedTime(&rtn.pbm, start_pbm, start_free);
    cudaEventElapsedTime(&rtn.freeCurand, start_free, stop_overall);

    return rtn;
}

#endif //__CoreModel_cuh__