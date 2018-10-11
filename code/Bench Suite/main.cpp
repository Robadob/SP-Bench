#define _CRT_SECURE_NO_WARNINGS
#define _2D
#include <cstdio>
#include <memory>
#include <string>
#include "results.h"
#include <ctime>
#ifdef _MSC_VER
#include <windows.h>
#define popen _popen
#define pclose _pclose
#else
//#include <filesystem>
#include <sys/stat.h>
#endif

CirclesParams reset(const CirclesParams &start, const CirclesParams &end);
NullParams reset(const NullParams &start, const NullParams &end);
DensityParams reset(const DensityParams &start, const DensityParams &end);
CirclesParams interpolateParams2D(const CirclesParams &start, const CirclesParams &end1, const CirclesParams &end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2);
NullParams interpolateParams2D(const NullParams &start, const NullParams &end1, const NullParams &end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2);
DensityParams interpolateParams2D(const DensityParams &start, const DensityParams &end1, const DensityParams &end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2);
NetworkParams interpolateParams2D(const NetworkParams &start, const NetworkParams &end1, const NetworkParams &end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2);
CirclesParams interpolateParams(const CirclesParams &start, const CirclesParams &end, const unsigned int step, const unsigned int totalSteps);
NullParams interpolateParams(const NullParams &start, const NullParams &end, const unsigned int step, const unsigned int totalSteps);
DensityParams interpolateParams(const DensityParams &start, const DensityParams &end, const unsigned int step, const unsigned int totalSteps);
NetworkParams interpolateParams(const NetworkParams &start, const NetworkParams &end, const unsigned int step, const unsigned int totalSteps);
template<class T>
bool executeBenchmark(const char* executable, T modelArgs, T *modelparamOut, unsigned int *agentCount, Time_Init *initRes, Time_Step_dbl *stepRes, float *totalTime, NeighbourhoodStats *nsFirst, NeighbourhoodStats *nsLast);
void logResult(FILE *out, const CirclesParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast);
void logHeader(FILE *out, const CirclesParams &modelArgs);
void logResult(FILE *out, const NullParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast);
void logHeader(FILE *out, const NullParams &modelArgs);
void logResult(FILE *out, const DensityParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast);
void logHeader(FILE *out, const DensityParams &modelArgs);
void logResult(FILE *out, const NetworkParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast);
void logHeader(FILE *out, const NetworkParams &modelArgs);
//Collated
void log(FILE *out, const NeighbourhoodStats *nsFirst);
void log(FILE *out, const CirclesParams *modelArgs);
void log(FILE *out, const NullParams *modelArgs);
void log(FILE *out, const DensityParams *modelArgs);
void log(FILE *out, const NetworkParams *modelArgs);
void log(FILE *out, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime);
void log(FILE *out, const unsigned int agentCount);

CirclesParams interpolateParams(const CirclesParams &start, const CirclesParams &end, const unsigned int step, const unsigned int totalSteps)
{
    const float stepsM1 = (float)totalSteps - 1.0f;
    CirclesParams modelArgs;
    modelArgs.seed = start.seed + (long long)((step / stepsM1)*((long long)end.seed - (long long)start.seed));
    //modelArgs.width = start.width + (int)((step / stepsM1)*((int)end.width - (int)start.width));
    modelArgs.agents = start.agents + (int)((step / stepsM1)*((int)end.agents - (int)start.agents));
    modelArgs.density = start.density + ((step / stepsM1)*(end.density - start.density));
    //modelArgs.interactionRad = start.interactionRad + ((step / stepsM1)*(end.interactionRad - start.interactionRad));
    //modelArgs.attractionForce = start.attractionForce + ((step / stepsM1)*(end.attractionForce - start.attractionForce));
    //modelArgs.repulsionForce = start.repulsionForce + ((step / stepsM1)*(end.repulsionForce - start.repulsionForce));
    modelArgs.forceModifier = start.forceModifier + ((step / stepsM1)*(end.forceModifier - start.forceModifier));
    modelArgs.iterations = start.iterations + (long long)((step / stepsM1)*((int)end.iterations - (int)start.iterations));
    return modelArgs;
}
CirclesParams reset(const CirclesParams &start, const CirclesParams &end)
{
    CirclesParams end1Reset;
    end1Reset.seed = end.seed - start.seed;
    //end1Reset.width = end.width - start.width;
    end1Reset.agents = end.agents - start.agents;
    end1Reset.density = end.density - start.density;
    //end1Reset.interactionRad = end.interactionRad - start.interactionRad;
    //end1Reset.attractionForce = end.attractionForce - start.attractionForce;
    //end1Reset.repulsionForce = end.repulsionForce - start.repulsionForce;
    end1Reset.forceModifier = end.forceModifier - start.forceModifier;
    end1Reset.iterations = end.iterations - start.iterations;
    return end1Reset;
}
NullParams reset(const NullParams &start, const NullParams &end)
{
    NullParams end1Reset;
    end1Reset.seed = end.seed - start.seed;
    end1Reset.agents = end.agents - start.agents;
    end1Reset.density = end.density - start.density;
    end1Reset.iterations = end.iterations - start.iterations;
    return end1Reset;
}
DensityParams reset(const DensityParams &start, const DensityParams &end)
{
    DensityParams end1Reset;
    end1Reset.seed = end.seed - start.seed;
    end1Reset.agentsPerCluster = end.agentsPerCluster - start.agentsPerCluster;
    end1Reset.envWidth = end.envWidth - start.envWidth;
    end1Reset.interactionRad = end.interactionRad - start.interactionRad;
    end1Reset.clusterCount = end.clusterCount - start.clusterCount;
    end1Reset.clusterRad = end.clusterRad - start.clusterRad;
    end1Reset.uniformDensity = end.uniformDensity - start.uniformDensity;
    end1Reset.iterations = end.iterations - start.iterations;
    return end1Reset;
}
NetworkParams reset(const NetworkParams &start, const NetworkParams &end)
{
    NetworkParams end1Reset;
    end1Reset.seed = end.seed - start.seed;
    end1Reset.agents = end.agents - start.agents;
    end1Reset.verts = end.verts - start.verts;
    end1Reset.edgesPerVert = end.edgesPerVert - start.edgesPerVert;
    end1Reset.capacityMod = end.capacityMod - start.capacityMod;
    end1Reset.iterations = end.iterations - start.iterations;
    return end1Reset;
}
CirclesParams interpolateParams2D(const CirclesParams &start, const CirclesParams &end1, const CirclesParams &end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2)
{
    CirclesParams end1Lerp = interpolateParams(CirclesParams::makeEmpty(), reset(start, end1), step1, totalSteps1);
    CirclesParams end2Lerp = interpolateParams(CirclesParams::makeEmpty(), reset(start, end2), step2, totalSteps2);
    CirclesParams modelArgs;
    modelArgs.seed = start.seed + end1Lerp.seed + end2Lerp.seed;
    //modelArgs.width = start.width + end1Lerp.width + end2Lerp.width;
    modelArgs.agents = start.agents + end1Lerp.agents + end2Lerp.agents;
    modelArgs.density = start.density + end1Lerp.density + end2Lerp.density;
    //modelArgs.interactionRad = start.interactionRad + end1Lerp.interactionRad + end2Lerp.interactionRad;
    //modelArgs.attractionForce = start.attractionForce + end1Lerp.attractionForce + end2Lerp.attractionForce;
    //modelArgs.repulsionForce = start.repulsionForce + end1Lerp.repulsionForce + end2Lerp.repulsionForce;
    modelArgs.forceModifier = start.forceModifier + end1Lerp.forceModifier + end2Lerp.forceModifier;
    modelArgs.iterations = start.iterations + end1Lerp.iterations + end2Lerp.iterations;
    return modelArgs;
}
NullParams interpolateParams2D(const NullParams &start, const NullParams &end1, const NullParams &end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2)
{
    NullParams end1Lerp = interpolateParams(NullParams::makeEmpty(), reset(start, end1), step1, totalSteps1);
    NullParams end2Lerp = interpolateParams(NullParams::makeEmpty(), reset(start, end2), step2, totalSteps2);
    NullParams modelArgs;
    modelArgs.seed = start.seed + end1Lerp.seed + end2Lerp.seed;
    modelArgs.agents = start.agents + end1Lerp.agents + end2Lerp.agents;
    modelArgs.density = start.density + end1Lerp.density + end2Lerp.density;
    modelArgs.iterations = start.iterations + end1Lerp.iterations + end2Lerp.iterations;
    return modelArgs;
}
DensityParams interpolateParams2D(const DensityParams &start, const DensityParams &end1, const DensityParams &end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2)
{
    DensityParams end1Lerp = interpolateParams(DensityParams::makeEmpty(), reset(start, end1), step1, totalSteps1);
    DensityParams end2Lerp = interpolateParams(DensityParams::makeEmpty(), reset(start, end2), step2, totalSteps2);
    DensityParams modelArgs;
    modelArgs.seed = start.seed + end1Lerp.seed + end2Lerp.seed;
    modelArgs.agentsPerCluster = start.agentsPerCluster + end1Lerp.agentsPerCluster + end2Lerp.agentsPerCluster;
    modelArgs.envWidth = start.envWidth + end1Lerp.envWidth + end2Lerp.envWidth;
    modelArgs.interactionRad = start.interactionRad + end1Lerp.interactionRad + end2Lerp.interactionRad;
    modelArgs.clusterCount = start.clusterCount + end1Lerp.clusterCount + end2Lerp.clusterCount;
    modelArgs.clusterRad = start.clusterRad + end1Lerp.clusterRad + end2Lerp.clusterRad;
    modelArgs.uniformDensity = start.uniformDensity + end1Lerp.uniformDensity + end2Lerp.uniformDensity;
    modelArgs.iterations = start.iterations + end1Lerp.iterations + end2Lerp.iterations;
    return modelArgs;
}
NetworkParams interpolateParams2D(const NetworkParams &start, const NetworkParams &end1, const NetworkParams &end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2)
{
    NetworkParams end1Lerp = interpolateParams(NetworkParams::makeEmpty(), reset(start, end1), step1, totalSteps1);
    NetworkParams end2Lerp = interpolateParams(NetworkParams::makeEmpty(), reset(start, end2), step2, totalSteps2);
    NetworkParams modelArgs;
    modelArgs.seed = start.seed + end1Lerp.seed + end2Lerp.seed;
    modelArgs.agents = start.agents + end1Lerp.agents + end2Lerp.agents;
    modelArgs.verts = start.verts + end1Lerp.verts + end2Lerp.verts;
    modelArgs.edgesPerVert = start.edgesPerVert + end1Lerp.edgesPerVert + end2Lerp.edgesPerVert;
    modelArgs.capacityMod = start.capacityMod + end1Lerp.capacityMod + end2Lerp.capacityMod;
    modelArgs.iterations = start.iterations + end1Lerp.iterations + end2Lerp.iterations;
    return modelArgs;
}
NullParams interpolateParams(const NullParams &start, const NullParams &end, const unsigned int step, const unsigned int totalSteps)
{
    const float stepsM1 = (float)totalSteps - 1.0f;
    NullParams modelArgs;
    modelArgs.seed = start.seed + (long long)((step / stepsM1)*((long long)end.seed - (long long)start.seed));
    modelArgs.agents = start.agents + (unsigned int)((step / stepsM1)*((unsigned int)end.agents - (unsigned int)start.agents));
    modelArgs.density = start.density + ((step / stepsM1)*(end.density - start.density));
    modelArgs.iterations = start.iterations + (long long)((step / stepsM1)*((int)end.iterations - (int)start.iterations));
    return modelArgs;
}
DensityParams interpolateParams(const DensityParams &start, const DensityParams &end, const unsigned int step, const unsigned int totalSteps)
{
    const float stepsM1 = (float)totalSteps - 1.0f;
    DensityParams modelArgs;
    modelArgs.seed = start.seed + (long long)((step / stepsM1)*((long long)end.seed - (long long)start.seed));
    modelArgs.agentsPerCluster = start.agentsPerCluster + (unsigned int)((step / stepsM1)*((unsigned int)end.agentsPerCluster - (unsigned int)start.agentsPerCluster));
    modelArgs.envWidth = start.envWidth + ((step / stepsM1)*(end.envWidth - start.envWidth));
    modelArgs.interactionRad = start.interactionRad + ((step / stepsM1)*(end.interactionRad - start.interactionRad));
    modelArgs.clusterCount = start.clusterCount + (unsigned int)((step / stepsM1)*((int)end.clusterCount - (int)start.clusterCount));
    modelArgs.clusterRad = start.clusterRad + ((step / stepsM1)*(end.clusterRad - start.clusterRad));
    modelArgs.uniformDensity = start.uniformDensity + ((step / stepsM1)*(end.uniformDensity - start.uniformDensity));
    modelArgs.iterations = start.iterations + (long long)((step / stepsM1)*((int)end.iterations - (int)start.iterations));
    return modelArgs;
}
NetworkParams interpolateParams(const NetworkParams &start, const NetworkParams &end, const unsigned int step, const unsigned int totalSteps)
{
    const float stepsM1 = (float)totalSteps - 1.0f;
    NetworkParams modelArgs;
    modelArgs.seed = start.seed + (long long)((step / stepsM1)*((long long)end.seed - (long long)start.seed));
    modelArgs.agents = start.agents + (unsigned int)((step / stepsM1)*((unsigned int)end.agents - (unsigned int)start.agents));
    modelArgs.verts = start.verts + (unsigned int)((step / stepsM1)*((unsigned int)end.verts - (unsigned int)start.verts));
    modelArgs.edgesPerVert = start.edgesPerVert + (unsigned int)((step / stepsM1)*((unsigned int)end.edgesPerVert - (unsigned int)start.edgesPerVert));
    modelArgs.capacityMod = start.capacityMod + ((step / stepsM1)*(end.capacityMod - start.capacityMod));
    modelArgs.iterations = start.iterations + (long long)((step / stepsM1)*((int)end.iterations - (int)start.iterations));
    return modelArgs;
}

template<class T>
bool executeBenchmark(const char* executable, T modelArgs, T *modelparamOut, unsigned int *agentCount, Time_Init *initRes, Time_Step_dbl *stepRes, float *totalTime, NeighbourhoodStats *nsFirst, NeighbourhoodStats *nsLast)
{
    char *command;
    bool rtn = true;
    ParamSet::execString(executable, modelArgs, &command);
#ifdef _MSC_VER
    std::shared_ptr<FILE> pipe(popen(command, "rb"), pclose);
#else
    std::shared_ptr<FILE> pipe(popen(command, "r"), pclose);
#endif
    if (!pipe.get()) rtn = false;
    if (rtn)
    {
        if (fread(modelparamOut, sizeof(T), 1, pipe.get()) != 1)
        {
            rtn = false; printf("\nReading model params failed.\n");
        }
        if (fread(agentCount, sizeof(unsigned int), 1, pipe.get()) != 1)
        {
            rtn = false; printf("Reading agent count failed.\n");
        }
        if (fread(initRes, sizeof(Time_Init), 1, pipe.get()) != 1)
        {
            rtn = false; printf("Reading init timings failed.\n");
        }
        if (fread(stepRes, sizeof(Time_Step_dbl), 1, pipe.get()) != 1)
        {
            rtn = false; printf("Reading step timings failed.\n");
        }
        if (fread(totalTime, sizeof(float), 1, pipe.get()) != 1)
        {
            rtn = false; printf("Reading total time failed.\n");
        }
        if (fread(nsFirst, sizeof(NeighbourhoodStats), 1, pipe.get()) != 1)
        {
            rtn = false; printf("Reading first NeighbourhoodStats failed.\n");
        }
        if (fread(nsLast, sizeof(NeighbourhoodStats), 1, pipe.get()) != 1)
        {
            rtn = false; printf("Reading last NeighbourhoodStats failed.\n");
        }
    }
    if (!rtn)
    {
        printf("Exec: %s\n", command);
    }
    return rtn;
}

/**
 * Logging
 */
void logHeader(FILE *out, const CirclesParams &modelArgs)
{
	fputs("model", out);
    fputs(",,,,,", out);
    //fputs(",,,,,,,", out);
	fputs("init (s)", out);
	fputs(",,,,,,", out);
	fputs("step avg (s)", out);
	fputs(",,,", out);
	fputs("overall (s)", out);
    fputs(",", out);
    fputs("Neighbourhood Stats", out);
    fputs(",,,,,,,,", out);
	fputs("\n", out);
	//ModelArgs
    //fputs("width", out);
    fputs("agents", out);
	fputs(",", out);
	fputs("density", out);
	fputs(",", out);
	//fputs("interactionRad", out);
	//fputs(",", out);
	//fputs("attractionForce", out);
	//fputs(",", out);
    //fputs("repulsionForce", out);
    fputs("forceModifier", out);
	fputs(",", out);
	fputs("iterations", out);
    fputs(",", out);
    fputs("seed", out);
    fputs(",", out);
	fputs("agents-out", out);
    fputs(",", out);
	//Init
	fputs("overall", out);
	fputs(",", out);
	fputs("initCurand", out);
	fputs(",", out);
	fputs("kernel", out);
	fputs(",", out);
	fputs("pbm", out);
	fputs(",", out);
	fputs("freeCurand", out);
	fputs(",", out);
	//Step avg
	fputs("overall", out);
	fputs(",", out);
	fputs("kernel", out);
	fputs(",", out);
	fputs("texture", out);
	fputs(",", out);
    //PBM Time
    fputs("PBMsort", out);
    fputs(",", out);
    fputs("PBMreorder", out);
    fputs(",", out);
    fputs("PBMtexcopy", out);
    fputs(",", out);
	//Total
	fputs("time", out);
    fputs(",", out);
    //Neighbourhood stats
    fputs("First Min", out);
    fputs(",", out);
    fputs("First Max", out);
    fputs(",", out);
    fputs("First Avg", out);
    fputs(",", out);
    fputs("First SD", out);
    fputs(",", out);
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
    fputs(",", out);
    fputs("Last SD", out);
    fputs(",", out);
	//ln
	fputs("\n", out);
}
void logResult(FILE *out, const CirclesParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast)
{	//ModelArgs
	//fprintf(out, "%u,%f,%f,%f,%f,%llu,%llu,",
	//	modelArgs->width,
	//	modelArgs->density,
	//	modelArgs->interactionRad,
	//	modelArgs->attractionForce,
	//	modelArgs->repulsionForce,
	//	modelArgs->iterations,
 //       modelArgs->seed
	//	);
    fprintf(out, "%u,%f,%f,%llu,%llu,",
        modelArgs->agents,
        modelArgs->density,
        modelArgs->forceModifier,
        modelArgs->iterations,
        modelArgs->seed
        );
	//Agent count
	fprintf(out, "%i,",
		agentCount
		);
	//Init
	fprintf(out, "%f,%f,%f,%f,%f,",
		initRes->overall/1000,
		initRes->initCurand / 1000,
		initRes->kernel / 1000,
		initRes->pbm / 1000,
		initRes->freeCurand / 1000
		);
	//Step avg
	fprintf(out, "%f,%f,%f,",
		stepRes->overall / 1000,
		stepRes->kernel / 1000,
		stepRes->texture / 1000
        );
    //PBM Time
    fprintf(out, "%f,%f,%f,",
        stepRes->pbm.sort / 1000,
        stepRes->pbm.reorder / 1000,
        stepRes->pbm.texcopy / 1000
        );
	//Total
	fprintf(out, "%f,",
		totalTime /1000
        );
    //Neighbourhood stats
    fprintf(out, "%d,%d,%f,%f,%d,%d,%f,%f,",
        nsFirst->min, nsFirst->max, nsFirst->average, nsFirst->standardDeviation,
        nsLast->min, nsLast->max, nsLast->average, nsLast->standardDeviation
        );
	//ln
	fputs("\n", out);
	fflush(out);
}
void logHeader(FILE *out, const NullParams &modelArgs)
{
    fputs("model", out);
    fputs(",,,,,", out);
    fputs("init (s)", out);
    fputs(",,,,,", out);
    fputs("step avg (s)", out);
    fputs(",,,", out);
    fputs("overall (s)", out);
    fputs(",", out);
    fputs("Neighbourhood Stats", out);
    fputs(",,,,,,,,", out);
    fputs("\n", out);
    //ModelArgs
    fputs("agents-in", out);
    fputs(",", out);
    fputs("density", out);
    fputs(",", out);
    fputs("iterations", out);
    fputs(",", out);
    fputs("seed", out);
    fputs(",", out);
    fputs("agents-out", out);
    fputs(",", out);
    //Init
    fputs("overall", out);
    fputs(",", out);
    fputs("initCurand", out);
    fputs(",", out);
    fputs("kernel", out);
    fputs(",", out);
    fputs("pbm", out);
    fputs(",", out);
    fputs("freeCurand", out);
    fputs(",", out);
    //Step avg
    fputs("overall", out);
    fputs(",", out);
    fputs("kernel", out);
    fputs(",", out);
    fputs("texture", out);
    fputs(",", out);
    //PBM Time
    fputs("PBMsort", out);
    fputs(",", out);
    fputs("PBMreorder", out);
    fputs(",", out);
    fputs("PBMtexcopy", out);
    fputs(",", out);
    //Total
    fputs("time", out);
    fputs(",", out);
    //Neighbourhood stats
    fputs("First Min", out);
    fputs(",", out);
    fputs("First Max", out);
    fputs(",", out);
    fputs("First Avg", out);
    fputs(",", out);
    fputs("First SD", out);
    fputs(",", out);
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
    fputs(",", out);
    fputs("Last SD", out);
    fputs(",", out);
    //ln
    fputs("\n", out);
}
void logResult(FILE *out, const NullParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast)
{	//ModelArgs
    fprintf(out, "%u,%f,%llu,%llu,",
        modelArgs->agents,
        modelArgs->density,
        modelArgs->iterations,
        modelArgs->seed
        );
    //Agent count
    fprintf(out, "%i,",
        agentCount
        );
    //Init
    fprintf(out, "%f,%f,%f,%f,%f,",
        initRes->overall / 1000,
        initRes->initCurand / 1000,
        initRes->kernel / 1000,
        initRes->pbm / 1000,
        initRes->freeCurand / 1000
        );
    //Step avg
    fprintf(out, "%f,%f,%f,",
        stepRes->overall / 1000,
        stepRes->kernel / 1000,
        stepRes->texture / 1000
        );
    //PBM Time
    fprintf(out, "%f,%f,%f,",
        stepRes->pbm.sort / 1000,
        stepRes->pbm.reorder / 1000,
        stepRes->pbm.texcopy / 1000
        );
    //Total
    fprintf(out, "%f,",
        totalTime / 1000
        );
    //Neighbourhood stats
    fprintf(out, "%d,%d,%f,%f,%d,%d,%f,%f,",
        nsFirst->min, nsFirst->max, nsFirst->average, nsFirst->standardDeviation,
        nsLast->min, nsLast->max, nsLast->average, nsLast->standardDeviation
        );
    //ln
    fputs("\n", out);
    fflush(out);
}
void logHeader(FILE *out, const DensityParams &modelArgs)
{
    fputs("model", out);
    fputs(",,,,,,,,", out);
    fputs("init (s)", out);
    fputs(",,,,,", out);
    fputs("step avg (s)", out);
    fputs(",,,", out);
    fputs("overall (s)", out);
    fputs(",", out);
    fputs("Neighbourhood Stats", out);
    fputs(",,,,,,,,", out);
    fputs("\n", out);
    //ModelArgs
    fputs("agents per cluster", out);
    fputs(",", out);
    fputs("envWidth", out);
    fputs(",", out);
    fputs("interactionRad", out);
    fputs(",", out);
    fputs("clusterCount", out);
    fputs(",", out);
    fputs("clusterRad", out);
    fputs(",", out);
    fputs("uniformDensity", out);
    fputs(",", out);
    fputs("iterations", out);
    fputs(",", out);
    fputs("seed", out);
    fputs(",", out);
    fputs("agents-out", out);
    fputs(",", out);
    //Init
    fputs("overall", out);
    fputs(",", out);
    fputs("initCurand", out);
    fputs(",", out);
    fputs("kernel", out);
    fputs(",", out);
    fputs("pbm", out);
    fputs(",", out);
    fputs("freeCurand", out);
    fputs(",", out);
    //Step avg
    fputs("overall", out);
    fputs(",", out);
    fputs("kernel", out);
    fputs(",", out);
    fputs("texture", out);
    fputs(",", out);
    //PBM Time
    fputs("PBMsort", out);
    fputs(",", out);
    fputs("PBMreorder", out);
    fputs(",", out);
    fputs("PBMtexcopy", out);
    fputs(",", out);
    //Total
    fputs("time", out);
    fputs(",", out);
    //Neighbourhood stats
    fputs("First Min", out);
    fputs(",", out);
    fputs("First Max", out);
    fputs(",", out);
    fputs("First Avg", out);
    fputs(",", out);
    fputs("First SD", out);
    fputs(",", out);
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
    fputs(",", out);
    fputs("Last SD", out);
    fputs(",", out);
    //ln
    fputs("\n", out);
}
void logResult(FILE *out, const DensityParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast)
{	//ModelArgs
    fprintf(out, "%u,%f,%f,%u,%f,%f,%llu,%llu,",
        modelArgs->agentsPerCluster,
        modelArgs->envWidth,
        modelArgs->interactionRad,
        modelArgs->clusterCount,
        modelArgs->clusterRad,
        modelArgs->uniformDensity,
        modelArgs->iterations,
        modelArgs->seed
        );
    //Agent count
    fprintf(out, "%i,",
        agentCount
        );
    //Init
    fprintf(out, "%f,%f,%f,%f,%f,",
        initRes->overall / 1000,
        initRes->initCurand / 1000,
        initRes->kernel / 1000,
        initRes->pbm / 1000,
        initRes->freeCurand / 1000
        );
    //Step avg
    fprintf(out, "%f,%f,%f,",
        stepRes->overall / 1000,
        stepRes->kernel / 1000,
        stepRes->texture / 1000
        );
    //PBM Time
    fprintf(out, "%f,%f,%f,",
        stepRes->pbm.sort / 1000,
        stepRes->pbm.reorder / 1000,
        stepRes->pbm.texcopy / 1000
        );
    //Total
    fprintf(out, "%f,",
        totalTime / 1000
        );
    //Neighbourhood stats
    fprintf(out, "%d,%d,%f,%f,%d,%d,%f,%f,",
        nsFirst->min, nsFirst->max, nsFirst->average, nsFirst->standardDeviation,
        nsLast->min, nsLast->max, nsLast->average, nsLast->standardDeviation
        );
    //ln
    fputs("\n", out);
    fflush(out);
}
void logHeader(FILE *out, const NetworkParams &modelArgs)
{
    fputs("model", out);
    fputs(",,,,,,", out);
    fputs("init (s)", out);
    fputs(",,,,,", out);
    fputs("step avg (s)", out);
    fputs(",,,", out);
    fputs("overall (s)", out);
    fputs(",", out);
    fputs("Neighbourhood Stats", out);
    fputs(",,,,,,,,", out);
    fputs("\n", out);
    //ModelArgs
    fputs("agents per cluster", out);
    fputs(",", out);
    fputs("envWidth", out);
    fputs(",", out);
    fputs("interactionRad", out);
    fputs(",", out);
    fputs("clusterCount", out);
    fputs(",", out);
    fputs("clusterRad", out);
    fputs(",", out);
    fputs("uniformDensity", out);
    fputs(",", out);
    fputs("iterations", out);
    fputs(",", out);
    fputs("seed", out);
    fputs(",", out);
    fputs("agents-out", out);
    fputs(",", out);
    //Init
    fputs("overall", out);
    fputs(",", out);
    fputs("initCurand", out);
    fputs(",", out);
    fputs("kernel", out);
    fputs(",", out);
    fputs("pbm", out);
    fputs(",", out);
    fputs("freeCurand", out);
    fputs(",", out);
    //Step avg
    fputs("overall", out);
    fputs(",", out);
    fputs("kernel", out);
    fputs(",", out);
    fputs("texture", out);
    fputs(",", out);
    //PBM Time
    fputs("PBMsort", out);
    fputs(",", out);
    fputs("PBMreorder", out);
    fputs(",", out);
    fputs("PBMtexcopy", out);
    fputs(",", out);
    //Total
    fputs("time", out);
    fputs(",", out);
    //Neighbourhood stats
    fputs("First Min", out);
    fputs(",", out);
    fputs("First Max", out);
    fputs(",", out);
    fputs("First Avg", out);
    fputs(",", out);
    fputs("First SD", out);
    fputs(",", out);
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
    fputs(",", out);
    fputs("Last SD", out);
    fputs(",", out);
    //ln
    fputs("\n", out);
}
void logResult(FILE *out, const NetworkParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast)
{	//ModelArgs
    fprintf(out, "%u,%u,%u,%f,%llu,%llu,",
        modelArgs->agents,
        modelArgs->verts,
        modelArgs->edgesPerVert,
        modelArgs->capacityMod,
        modelArgs->iterations,
        modelArgs->seed
        );
    //Agent count
    fprintf(out, "%i,",
        agentCount
        );
    //Init
    fprintf(out, "%f,%f,%f,%f,%f,",
        initRes->overall / 1000,
        initRes->initCurand / 1000,
        initRes->kernel / 1000,
        initRes->pbm / 1000,
        initRes->freeCurand / 1000
        );
    //Step avg
    fprintf(out, "%f,%f,%f,",
        stepRes->overall / 1000,
        stepRes->kernel / 1000,
        stepRes->texture / 1000
        );
    //PBM Time
    fprintf(out, "%f,%f,%f,",
        stepRes->pbm.sort / 1000,
        stepRes->pbm.reorder / 1000,
        stepRes->pbm.texcopy / 1000
        );
    //Total
    fprintf(out, "%f,",
        totalTime / 1000
        );
    //Neighbourhood stats
    fprintf(out, "%d,%d,%f,%f,%d,%d,%f,%f,",
        nsFirst->min, nsFirst->max, nsFirst->average, nsFirst->standardDeviation,
        nsLast->min, nsLast->max, nsLast->average, nsLast->standardDeviation
        );
    //ln
    fputs("\n", out);
    fflush(out);
}

void log(FILE *out, const NeighbourhoodStats *ns)
{
    //Neighbourhood stats
    fprintf(out, "%d,%d,%f,%f,",
        ns->min, ns->max, ns->average, ns->standardDeviation
        );
}
void log(FILE *out, const CirclesParams *modelArgs)
{
    //ModelArgs
    //fprintf(out, "%u,%f,%f,%f,%f,%llu,%llu,",
    //    modelArgs->width,
    //    modelArgs->density,
    //    modelArgs->interactionRad,
    //    modelArgs->attractionForce,
    //    modelArgs->repulsionForce,
    //    modelArgs->iterations,
    //    modelArgs->seed
    //    );
    fprintf(out, "%u,%f,%f,%llu,%llu,",
        modelArgs->agents,
        modelArgs->density,
        modelArgs->forceModifier,
        modelArgs->iterations,
        modelArgs->seed
        );
}
void log(FILE *out, const NullParams *modelArgs)
{	
    //ModelArgs
    fprintf(out, "%u,%f,%llu,%llu,",
        modelArgs->agents,
        modelArgs->density,
        modelArgs->iterations,
        modelArgs->seed
        );
}
void log(FILE *out, const DensityParams *modelArgs)
{	
    //ModelArgs
    fprintf(out, "%u,%f,%f,%u,%f,%f,%llu,%llu,",
        modelArgs->agentsPerCluster,
        modelArgs->envWidth,
        modelArgs->interactionRad,
        modelArgs->clusterCount,
        modelArgs->clusterRad,
        modelArgs->uniformDensity,
        modelArgs->iterations,
        modelArgs->seed
        );
}
void log(FILE *out, const NetworkParams *modelArgs)
{
    //ModelArgs
    fprintf(out, "%u,%u,%u,%f,%llu,%llu,",
        modelArgs->agents,
        modelArgs->verts,
        modelArgs->edgesPerVert,
        modelArgs->capacityMod,
        modelArgs->iterations,
        modelArgs->seed
        );
}
void log(FILE *out, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime)
{
    //Init
    fprintf(out, "%f,%f,%f,%f,%f,",
        initRes->overall / 1000,
        initRes->initCurand / 1000,
        initRes->kernel / 1000,
        initRes->pbm / 1000,
        initRes->freeCurand / 1000
        );
    //Step avg
    fprintf(out, "%f,%f,%f,",
        stepRes->overall / 1000,
        stepRes->kernel / 1000,
        stepRes->texture / 1000
        );
    //PBM Time
    fprintf(out, "%f,%f,%f,",
        stepRes->pbm.sort / 1000,
        stepRes->pbm.reorder / 1000,
        stepRes->pbm.texcopy / 1000
        );
    //Total
    fprintf(out, "%f,",
        totalTime / 1000
        );
}
void log(FILE *out, const unsigned int agentCount)
{
    fprintf(out, "%u,", agentCount);
}