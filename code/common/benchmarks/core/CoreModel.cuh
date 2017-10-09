#ifndef __CoreModel_cuh__
#define __CoreModel_cuh__

#include "CoreKernels.cuh"

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
        init_particles << <initBlocks, initThreads >> >(d_rng, d_lm);//Defined in CircleKernels.cuh
    else
    {
        init_particles_uniform << <initBlocks, initThreads >> >(d_lm);//Defined in CircleKernels.cuh
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

#endif //__CoreModel_cuh__