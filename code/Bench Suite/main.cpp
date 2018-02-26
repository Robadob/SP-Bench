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
CirclesParams interpolateParams(const CirclesParams &start, const CirclesParams &end, const unsigned int step, const unsigned int totalSteps);
NullParams interpolateParams(const NullParams &start, const NullParams &end, const unsigned int step, const unsigned int totalSteps);
DensityParams interpolateParams(const DensityParams &start, const DensityParams &end, const unsigned int step, const unsigned int totalSteps);
template<class T>
bool run(const T &start, const T &end, const unsigned int steps, const char *runName);
template<class T>
bool executeBenchmark(const char* executable, T modelArgs, T *modelparamOut, unsigned int *agentCount, Time_Init *initRes, Time_Step_dbl *stepRes, float *totalTime, NeighbourhoodStats *nsFirst, NeighbourhoodStats *nsLast);
void logResult(FILE *out, const CirclesParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast);
void logHeader(FILE *out, const CirclesParams &modelArgs);
void logResult(FILE *out, const NullParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast);
void logHeader(FILE *out, const NullParams &modelArgs);
void logResult(FILE *out, const DensityParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime, const NeighbourhoodStats *nsFirst, const NeighbourhoodStats *nsLast);
void logHeader(FILE *out, const DensityParams &modelArgs);
//Collated
template<class T>
bool runCollated(const T &start, const T &end, const unsigned int steps, const char *runName);
template<class T>
bool runCollated2D(const T &start, const T &end1, const T &end2, const unsigned int steps1, const unsigned int steps2, const char *runName);
void log(FILE *out, const NeighbourhoodStats *nsFirst);
void log(FILE *out, const CirclesParams *modelArgs);
void log(FILE *out, const NullParams *modelArgs);
void log(FILE *out, const DensityParams *modelArgs);
void log(FILE *out, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime);
void log(FILE *out, const unsigned int agentCount);

template<class T>
bool runCollated(const T &start, const T &end, const unsigned int steps, const char *runName)
{
    printf("Running %s\n", runName);
    const float stepsM1 = (float)steps - 1.0f;
    //Create objects for use within the loop
    Time_Init initRes;
    Time_Step_dbl stepRes;
    NeighbourhoodStats nsFirst, nsLast;
    T modelArgs;
    T modelParamsOut;
    unsigned int agentCount;
    float totalTime;
    //Create log
    FILE *log_F = nullptr;
    {
        std::string logPath("./out/");
#ifdef _MSC_VER
        CreateDirectory(logPath.c_str(), NULL);
#else
        mkdir(logPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
        logPath = logPath.append(runName);
        logPath = logPath.append("/");
#ifdef _MSC_VER
        CreateDirectory(logPath.c_str(), NULL);
#else
        mkdir(logPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        //if (!exists(logPath)) { // Check if folder exists
        //    create_directory(logPath.c_str()); // create folder
        //}
#endif
        logPath = logPath.append("collated");
        logPath = logPath.append(std::to_string(time(0)));
        logPath = logPath.append(".csv");
        log_F = fopen(logPath.c_str(), "w");
        if (!log_F)
        {
            fprintf(stderr, "Collated benchmark '%s' failed to create log file '%s'\n", runName, logPath.c_str());
            return false;
        }
    }
    //Create log header
    logCollatedHeader(log_F, start);
    //For each step
    for (unsigned int i = 0; i < steps; i++)
    {
        //Interpolate model
        modelArgs = interpolateParams(start, end, i, steps);
        //Clear output structures
        memset(&modelParamsOut, 0, sizeof(T));
        memset(&stepRes, 0, sizeof(Time_Step_dbl));
        memset(&initRes, 0, sizeof(Time_Init));
        //For each mod
        for (unsigned int j = 0; j < sizeof(TEST_EXECUTABLES) / sizeof(char*); ++j)
        {
            printf("\rExecuting run %s %i/%i %i/%llu      ", runName, i + 1, (int)steps, j + 1, sizeof(TEST_EXECUTABLES) / sizeof(char*));
            agentCount = 0;
            totalTime = 0;
            //executeBenchmark
            if (!executeBenchmark(TEST_EXECUTABLES[j], modelArgs, &modelParamsOut, &agentCount, &initRes, &stepRes, &totalTime, &nsFirst, &nsLast))
            {
                fprintf(stderr, "\rBenchmark '%s' '%s' execution failed on stage %d/%d, exiting early.\n", runName, TEST_NAMES[j], i + 1, steps);
                return false;
            }
            //logResult
            log(log_F, &initRes, &stepRes, totalTime);
        }
        //AgentCount
        log(log_F, agentCount);
        //Neighbourhood stats
        log(log_F, &nsFirst);
        log(log_F, &nsLast);
        //Model Args
        log(log_F, &modelArgs);
        fputs("\n", log_F);
        fflush(log_F);
    }
    //Close log
    fclose(log_F);
    //Print confirmation to console
    printf("\rCompleted run %s %i/%i %llu/%llu         \n", runName, (int)steps, (int)steps, sizeof(TEST_EXECUTABLES) / sizeof(char*), sizeof(TEST_EXECUTABLES) / sizeof(char*));
    return true;
}
template<class T>
bool runCollated2D(const T &start, const T &end1, const T &end2, const unsigned int steps1, const unsigned int steps2, const char *runName)
{
    printf("Running %s\n", runName);
    //Create objects for use within the loop
    Time_Init initRes;
    Time_Step_dbl stepRes;
    NeighbourhoodStats nsFirst, nsLast;
    T modelArgs;
    T modelParamsOut;
    unsigned int agentCount;
    float totalTime;
    //Create log
    FILE *log_F = nullptr;
    {
        std::string logPath("./out/");
#ifdef _MSC_VER
        CreateDirectory(logPath.c_str(), NULL);
#else
        mkdir(logPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
        logPath = logPath.append(runName);
        logPath = logPath.append("/");
#ifdef _MSC_VER
        CreateDirectory(logPath.c_str(), NULL);
#else
        mkdir(logPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        //if (!exists(logPath)) { // Check if folder exists
        //    create_directory(logPath.c_str()); // create folder
        //}
#endif
        logPath = logPath.append("collated");
        logPath = logPath.append(std::to_string(time(0)));
        logPath = logPath.append(".csv");
        log_F = fopen(logPath.c_str(), "w");
        if (!log_F)
        {
            fprintf(stderr, "Collated benchmark '%s' failed to create log file '%s'\n", runName, logPath.c_str());
            return false;
        }
    }
    //Create log header
    logCollatedHeader(log_F, start);
    //For each step
    for (unsigned int w = 0; w < steps1; w++)
    {
        //Interpolate model
        for (unsigned int v = 0; v < steps2; v++)
        {
            //Interpolate model
            modelArgs = interpolateParams2D(start, end1, end2, w, steps1, v, steps2);
            //Clear output structures
            memset(&modelParamsOut, 0, sizeof(T));
            memset(&stepRes, 0, sizeof(Time_Step_dbl));
            memset(&initRes, 0, sizeof(Time_Init));
            //For each mod
            for (unsigned int j = 0; j < sizeof(TEST_EXECUTABLES) / sizeof(char*); ++j)
            {
                printf("\rExecuting run %s %d:%d/%d:%d %i/%lu      ", runName, w + 1, v + 1, steps1, steps2, j + 1, sizeof(TEST_EXECUTABLES) / sizeof(char*));
                agentCount = 0;
                totalTime = 0;
                //executeBenchmark
                if (!executeBenchmark(TEST_EXECUTABLES[j], modelArgs, &modelParamsOut, &agentCount, &initRes, &stepRes, &totalTime, &nsFirst, &nsLast))
                {
                    fprintf(stderr, "\rBenchmark '%s' '%s' execution failed on stage %d:%d/%d:%d, exiting early.\n", runName, TEST_NAMES[j], w + 1, v + 1, steps1, steps2);
                    return false;
                }
                //logResult
                log(log_F, &initRes, &stepRes, totalTime);
            }
            //AgentCount
            log(log_F, agentCount);
            //Neighbourhood stats
            log(log_F, &nsFirst);
            log(log_F, &nsLast);
            //Model Args
            log(log_F, &modelArgs);
            fputs("\n", log_F);
            fflush(log_F);
        }
    }
    //Close log
    fclose(log_F);
    //Print confirmation to console
    printf("\rCompleted run %s %i/%i %lu/%lu         \n", runName, (int)steps1, (int)steps2, sizeof(TEST_EXECUTABLES) / sizeof(char*), sizeof(TEST_EXECUTABLES) / sizeof(char*));
    return true;
}
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

template<class T>
bool executeBenchmark(const char* executable, T modelArgs, T *modelparamOut, unsigned int *agentCount, Time_Init *initRes, Time_Step_dbl *stepRes, float *totalTime, NeighbourhoodStats *nsFirst, NeighbourhoodStats *nsLast)
{
    char *command;
    bool rtn = true;
    ParamSet::execString(executable, modelArgs, &command);
    std::shared_ptr<FILE> pipe(popen(command, "rb"), pclose);
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
    fputs(",,,,,,", out);
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
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
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
	//Total
	fprintf(out, "%f,",
		totalTime /1000
        );
    //Neighbourhood stats
    fprintf(out, "%d,%d,%f,%d,%d,%f,",
        nsFirst->min, nsFirst->max, nsFirst->average,
        nsLast->min, nsLast->max, nsLast->average
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
    fputs(",,,,,,", out);
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
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
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
    //Total
    fprintf(out, "%f,",
        totalTime / 1000
        );
    //Neighbourhood stats
    fprintf(out, "%d,%d,%f,%d,%d,%f,",
        nsFirst->min, nsFirst->max, nsFirst->average,
        nsLast->min, nsLast->max, nsLast->average
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
    fputs(",,,,,,", out);
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
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
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
    //Total
    fprintf(out, "%f,",
        totalTime / 1000
        );
    //Neighbourhood stats
    fprintf(out, "%d,%d,%f,%d,%d,%f,",
        nsFirst->min, nsFirst->max, nsFirst->average,
        nsLast->min, nsLast->max, nsLast->average
        );
    //ln
    fputs("\n", out);
    fflush(out);
}

void log(FILE *out, const NeighbourhoodStats *nsFirst)
{
    //Neighbourhood stats
    fprintf(out, "%d,%d,%f,",
        nsFirst->min, nsFirst->max, nsFirst->average
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
    //Total
    fprintf(out, "%f,",
        totalTime / 1000
        );
}
void log(FILE *out, const unsigned int agentCount)
{
    fprintf(out, "%u,", agentCount);
}