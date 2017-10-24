#define _CRT_SECURE_NO_WARNINGS
#define _2D
#include <cstdio>
#include <memory>
#include <string>
#include "results.h"
#include <ctime>
#ifdef _MSC_BUILD 
#include <windows.h>
#else
#include <filesystem>
#endif

CirclesParams interpolateParams(const CirclesParams &start, const CirclesParams &end, const unsigned int step, const unsigned int totalSteps);
NullParams interpolateParams(const NullParams &start, const NullParams &end, const unsigned int step, const unsigned int totalSteps);
DensityParams interpolateParams(const DensityParams &start, const DensityParams &end, const unsigned int step, const unsigned int totalSteps);
void execString(const char* executable, CirclesParams model, char **rtn);
void execString(const char* executable, NullParams model, char **rtn);
template<class T>
bool run(const T &start, const T &end, const unsigned int steps, const char *runName);
template<class T>
bool executeBenchmark(const char* executable, T modelArgs, T *modelparamOut, unsigned int *agentCount, Time_Init *initRes, Time_Step_dbl *stepRes, float *totalTime);
void logResult(FILE *out, const CirclesParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime);
void logHeader(FILE *out, const CirclesParams &modelArgs);
void logResult(FILE *out, const NullParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime);
void logHeader(FILE *out, const NullParams &modelArgs);
void logResult(FILE *out, const DensityParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime);
void logHeader(FILE *out, const DensityParams &modelArgs);
const char *TEST_NAMES[] = { "Default", "Strips", "Morton", "MortonCompute", "Hilbert", "Peano" };
const char *TEST_EXECUTABLES[] = { "Release-Mod-Default.exe", "Release-Mod-Strips.exe", "Release-Mod-Morton.exe", "Release-Mod-MortonCompute.exe", "Release-Mod-Hilbert.exe", "Release-Mod-Peano.exe" };
const char *DIR_X64 = "..\\bin\\x64\\";
int main(int argc, char* argv[])
{
#if defined(_3D)
    {//Problem Scale - Circles
        //Init step count
        const int steps = 11;        
        //Init model arg start
        CirclesParams start = {};
        start.iterations = 1000;
        start.density = 0.01f;
        start.interactionRad = 5.0f;
        start.width = 300;
        start.seed = 100;
        //Init model arg end
        CirclesParams end = {};
        end.iterations = 1000;
        end.density = 0.01f;
        end.interactionRad = 5.0f;
        end.width = 300;
        end.seed = 250;
        run(start, end, steps, "CirclesProblemScale");
    }
    {//Neighbourhood Scale - Circles
        const int steps = 16;        
        //Init model arg start
        CirclesParams  start = {};
        start.iterations = 1000;
        start.density = 0.01f;
        start.interactionRad = 1.0f;
        start.width = 100;
        //Init model arg end
        CirclesParams end = {};
        end.iterations = 1000;
        end.density = 0.01f;
        end.interactionRad = 15.0f;
        end.width = 100;
        //Init step count
        run(start, end, steps, "CirclesNeighbourhoodScale");
    }
#elif defined(_2D)
    {//Problem Scale - Circles
        //Init step count
        const int steps = 11;
        //Init model arg start
        CirclesParams start = {};
        start.iterations = 1000;
        start.density = 0.01f;
        start.interactionRad = 5.0f;
        start.width = 300;
        start.seed = 100;
        //Init model arg end
        CirclesParams end = {};
        end.iterations = 1000;
        end.density = 0.01f;
        end.interactionRad = 5.0f;
        end.width = 300;
        end.seed = 250;
        run(start, end, steps, "CirclesProblemScale");
    }
    {//Neighbourhood Scale - Circles
        const int steps = 16;
        //Init model arg start
        CirclesParams  start = {};
        start.iterations = 1000;
        start.density = 0.01f;
        start.interactionRad = 1.0f;
        start.width = 100;
        start.seed = 100;
        //Init model arg end
        CirclesParams end = {};
        end.iterations = 1000;
        end.density = 0.01f;
        end.interactionRad = 15.0f;
        end.width = 100;
        end.seed = 250;
        //Init step count
        run(start, end, steps, "CirclesNeighbourhoodScale");
    }
    {//Something Null
        const int steps = 16;
        ////Init model arg start
        NullParams  start = {};
        //start.iterations = 1000;
        //start.density = 0.01f;
        //start.interactionRad = 1.0f;
        //start.agents = 100;
        ////Init model arg end
        NullParams end = {};
        //end.iterations = 1000;
        //end.density = 0.01f;
        //end.interactionRad = 15.0f;
        //end.agents = 100;
        ////Init step count
        run(start, end, steps, "CirclesNeighbourhoodScale");
    }
    {//Something Density
        const int steps = 16;
        ////Init model arg start
        DensityParams  start = {};
        //start.iterations = 1000;
        //start.density = 0.01f;
        //start.interactionRad = 1.0f;
        //start.agents = 100;
        ////Init model arg end
        DensityParams end = {};
        //end.iterations = 1000;
        //end.density = 0.01f;
        //end.interactionRad = 15.0f;
        //end.agents = 100;
        ////Init step count
        run(start, end, steps, "CirclesNeighbourhoodScale");
    }
#endif
	printf("\nComplete\n");
}
template<class T>
bool run(const T &start, const T &end, const unsigned int steps, const char *runName)
{
    printf("Running %s\n", runName);
    const float stepsM1 = (float)steps - 1.0f;
    //Create objects for use within the loop
    Time_Init initRes;
    Time_Step_dbl stepRes;
    T modelArgs;
    T modelParamsOut;
    unsigned int agentCount;
    float totalTime;
    //For each mod
    for (unsigned int j = 0; j < sizeof(TEST_EXECUTABLES) / sizeof(char*); ++j)
    {
        //Create log
        std::string logPath("./out/");
#ifdef _MSC_BUILD
        CreateDirectory(logPath.c_str(), NULL);
#endif
        logPath = logPath.append(runName);
        logPath = logPath.append("/");
#ifdef _MSC_BUILD
        CreateDirectory(logPath.c_str(), NULL);
#endif
        logPath = logPath.append(TEST_NAMES[j]);
        logPath = logPath.append("/");
#ifdef _MSC_BUILD
        CreateDirectory(logPath.c_str(), NULL);
#endif
#ifndef _MSC_BUILD
        if (!exists(logPath)) { // Check if folder exists
            create_directory(logPath.c_str()); // create folder
        }
#endif
        logPath = logPath.append(std::to_string(time(0)));
        logPath = logPath.append(".csv");
        FILE *log = fopen(logPath.c_str(), "w");
        if (!log)
        {
            fprintf(stderr,"Failed to create log filed '%s'\n", logPath.c_str());
            return false;
        }
        //Create log header
        logHeader(log, start);
        //For each benchmark
        for (unsigned int i = 0; i < steps; i++)
        {
            printf("\rExecuting run %s %i/%i      ", TEST_NAMES[j], i, (int)stepsM1);
            //Interpolate model
            modelArgs = interpolateParams(start, end, i, steps);
            //Clear output structures
            memset(&modelParamsOut, 0, sizeof(T));
            memset(&stepRes, 0, sizeof(Time_Step_dbl));
            memset(&initRes, 0, sizeof(Time_Init));
            agentCount = 0;
            totalTime = 0;
            //executeBenchmark
            if (!executeBenchmark(TEST_EXECUTABLES[j], modelArgs, &modelParamsOut, &agentCount, &initRes, &stepRes, &totalTime))
            {
                fprintf(stderr, "\rBenchmark '%s' '%s' execution failed, exiting early.\n", runName, TEST_NAMES[j]);
                return false;
            }
            //logResult
            logResult(log, &modelParamsOut, agentCount, &initRes, &stepRes, totalTime);
        }
        printf("\n");
        //Close log
        fclose(log);
    }
    return true;
}
CirclesParams interpolateParams(const CirclesParams &start, const CirclesParams &end, const unsigned int step, const unsigned int totalSteps)
{
    const float stepsM1 = (float)totalSteps - 1.0f;
    CirclesParams modelArgs;
    modelArgs.seed = start.seed + (long long)((step / stepsM1)*((long long)end.seed - (long long)start.seed));
    modelArgs.width = start.width + (int)((step / stepsM1)*((int)end.width - (int)start.width));
    modelArgs.density = start.density + ((step / stepsM1)*(end.density - start.density));
    modelArgs.interactionRad = start.interactionRad + ((step / stepsM1)*(end.interactionRad - start.interactionRad));
    modelArgs.attractionForce = start.attractionForce + ((step / stepsM1)*(end.attractionForce - start.attractionForce));
    modelArgs.repulsionForce = start.repulsionForce + ((step / stepsM1)*(end.repulsionForce - start.repulsionForce));
    modelArgs.iterations = start.iterations + (long long)((step / stepsM1)*((int)end.iterations - (int)start.iterations));
    return modelArgs;
}
NullParams interpolateParams(const NullParams &start, const NullParams &end, const unsigned int step, const unsigned int totalSteps)
{
    const float stepsM1 = (float)totalSteps - 1.0f;
    NullParams modelArgs;
    modelArgs.seed = start.seed + (long long)((step / stepsM1)*((long long)end.seed - (long long)start.seed));
    modelArgs.agents = start.agents + (unsigned int)((step / stepsM1)*((unsigned int)end.agents - (unsigned int)start.agents));
    modelArgs.density = start.density + ((step / stepsM1)*(end.density - start.density));
    modelArgs.interactionRad = start.interactionRad + ((step / stepsM1)*(end.interactionRad - start.interactionRad));
    modelArgs.iterations = start.iterations + (long long)((step / stepsM1)*((int)end.iterations - (int)start.iterations));
    return modelArgs;
}
DensityParams interpolateParams(const DensityParams &start, const DensityParams &end, const unsigned int step, const unsigned int totalSteps)
{
    const float stepsM1 = (float)totalSteps - 1.0f;
    DensityParams modelArgs;
    modelArgs.seed = start.seed + (long long)((step / stepsM1)*((long long)end.seed - (long long)start.seed));
    modelArgs.agents = start.agents + (unsigned int)((step / stepsM1)*((unsigned int)end.agents - (unsigned int)start.agents));
    modelArgs.interactionRad = start.interactionRad + ((step / stepsM1)*(end.interactionRad - start.interactionRad));
    modelArgs.clusterCount = start.clusterCount + ((step / stepsM1)*(end.clusterCount - start.clusterCount));
    modelArgs.clusterRad = start.clusterRad + ((step / stepsM1)*(end.clusterRad - start.clusterRad));
    modelArgs.iterations = start.iterations + (long long)((step / stepsM1)*((int)end.iterations - (int)start.iterations));
    return modelArgs;
}
template<class T>
bool executeBenchmark(const char* executable, T modelArgs, T *modelparamOut, unsigned int *agentCount, Time_Init *initRes, Time_Step_dbl *stepRes, float *totalTime)
{
    char *command;
    bool rtn = true;
    execString(executable, modelArgs, &command);
    std::shared_ptr<FILE> pipe(_popen(command, "rb"), _pclose);
    if (!pipe.get()) return false;
    if (fread(modelparamOut, sizeof(T), 1, pipe.get()) != 1)
    {
        rtn = false; printf("\nReading model params failed.\n");
    }
    if (fread(agentCount, sizeof(unsigned int), 1, pipe.get()) != 1)
    {
        rtn = false; printf("\nReading agent count failed.\n");
    }
    if (fread(initRes, sizeof(Time_Init), 1, pipe.get()) != 1)
    {
        rtn = false; printf("\nReading init timings failed.\n");
    }
    if (fread(stepRes, sizeof(Time_Step_dbl), 1, pipe.get()) != 1)
    {
        rtn = false; printf("\nReading step timings failed.\n");
    }
    if (fread(totalTime, sizeof(float), 1, pipe.get()) != 1)
    {
        rtn = false; printf("\nReading total time failed.\n");
    }
    return rtn;
}

void execString(const char* executable, CirclesParams modelArgs, char **rtn)
{
	std::string buffer("\"");
	buffer = buffer.append(DIR_X64);
	buffer = buffer.append(executable);
	buffer = buffer.append("\"");
	buffer = buffer.append("-pipe");
	buffer = buffer.append(" ");
	buffer = buffer.append("-device");
	buffer = buffer.append(" ");
	buffer = buffer.append(std::to_string(0));
	buffer = buffer.append(" ");
	buffer = buffer.append("-circles");
	buffer = buffer.append(" ");
	buffer = buffer.append(std::to_string(modelArgs.width));
	buffer = buffer.append(" ");
	buffer = buffer.append(std::to_string(modelArgs.density));
	buffer = buffer.append(" ");
	buffer = buffer.append(std::to_string(modelArgs.interactionRad));
	buffer = buffer.append(" ");
	buffer = buffer.append(std::to_string(modelArgs.attractionForce));
	buffer = buffer.append(" ");
	buffer = buffer.append(std::to_string(modelArgs.repulsionForce));
	buffer = buffer.append(" ");
	buffer = buffer.append(std::to_string(modelArgs.iterations));
	if (modelArgs.seed!=12)
	{

		buffer = buffer.append(" ");
		buffer = buffer.append(" -seed");
		buffer = buffer.append(" ");
		buffer = buffer.append(std::to_string(modelArgs.seed));
	}
	const char *src = buffer.c_str();
	*rtn = (char *)malloc(sizeof(char*)*(buffer.length() + 1));
	memcpy(*rtn, src, sizeof(char*)*(buffer.length() + 1));
}
void execString(const char* executable, NullParams modelArgs, char **rtn)
{
    std::string buffer("\"");
    buffer = buffer.append(DIR_X64);
    buffer = buffer.append(executable);
    buffer = buffer.append("\"");
    buffer = buffer.append("-pipe");
    buffer = buffer.append(" ");
    buffer = buffer.append("-device");
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(0));
    buffer = buffer.append(" ");
    buffer = buffer.append("-null");
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.agents));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.density));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.interactionRad));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.iterations));
    if (modelArgs.seed != 12)
    {

        buffer = buffer.append(" ");
        buffer = buffer.append(" -seed");
        buffer = buffer.append(" ");
        buffer = buffer.append(std::to_string(modelArgs.seed));
    }
    const char *src = buffer.c_str();
    *rtn = (char *)malloc(sizeof(char*)*(buffer.length() + 1));
    memcpy(*rtn, src, sizeof(char*)*(buffer.length() + 1));
}
void execString(const char* executable, DensityParams modelArgs, char **rtn)
{
    std::string buffer("\"");
    buffer = buffer.append(DIR_X64);
    buffer = buffer.append(executable);
    buffer = buffer.append("\"");
    buffer = buffer.append("-pipe");
    buffer = buffer.append(" ");
    buffer = buffer.append("-device");
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(0));
    buffer = buffer.append(" ");
    buffer = buffer.append("-density");
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.agents));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.envWidth));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.clusterCount));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.clusterRad));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.interactionRad));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.iterations));
    if (modelArgs.seed != 12)
    {

        buffer = buffer.append(" ");
        buffer = buffer.append(" -seed");
        buffer = buffer.append(" ");
        buffer = buffer.append(std::to_string(modelArgs.seed));
    }
    const char *src = buffer.c_str();
    *rtn = (char *)malloc(sizeof(char*)*(buffer.length() + 1));
    memcpy(*rtn, src, sizeof(char*)*(buffer.length() + 1));
}
/**
 * Logging
 */
void logHeader(FILE *out, const CirclesParams &modelArgs)
{
	fputs("model", out);
	fputs(",,,,,,,", out);
	fputs("init (s)", out);
	fputs(",,,,,,", out);
	fputs("step avg (s)", out);
	fputs(",,,", out);
	fputs("overall (s)", out);
	fputs(",", out);
	fputs("\n", out);
	//ModelArgs
	fputs("width", out);
	fputs(",", out);
	fputs("density", out);
	fputs(",", out);
	fputs("interactionRad", out);
	fputs(",", out);
	fputs("attractionForce", out);
	fputs(",", out);
	fputs("repulsionForce", out);
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
	//ln
	fputs("\n", out);
}
void logResult(FILE *out, const CirclesParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime)
{	//ModelArgs
	fprintf(out, "%i,%f,%f,%f,%f,%llu,%llu,",
		modelArgs->width,
		modelArgs->density,
		modelArgs->interactionRad,
		modelArgs->attractionForce,
		modelArgs->repulsionForce,
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
	//ln
	fputs("\n", out);
	fflush(out);
}
void logHeader(FILE *out, const NullParams &modelArgs)
{
    fputs("model", out);
    fputs(",,,,,,", out);
    fputs("init (s)", out);
    fputs(",,,,,,", out);
    fputs("step avg (s)", out);
    fputs(",,,", out);
    fputs("overall (s)", out);
    fputs(",", out);
    fputs("\n", out);
    //ModelArgs
    fputs("agents-in", out);
    fputs(",", out);
    fputs("density", out);
    fputs(",", out);
    fputs("interactionRad", out);
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
    //ln
    fputs("\n", out);
}
void logResult(FILE *out, const NullParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime)
{	//ModelArgs
    fprintf(out, "%i,%f,%f,%llu,%llu,",
        modelArgs->agents,
        modelArgs->density,
        modelArgs->interactionRad,
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
    //ln
    fputs("\n", out);
    fflush(out);
}
void logHeader(FILE *out, const DensityParams &modelArgs)
{
    fputs("model", out);
    fputs(",,,,,,,", out);
    fputs("init (s)", out);
    fputs(",,,,,,", out);
    fputs("step avg (s)", out);
    fputs(",,,", out);
    fputs("overall (s)", out);
    fputs(",", out);
    fputs("\n", out);
    //ModelArgs
    fputs("agents-in", out);
    fputs(",", out);
    fputs("interactionRad", out);
    fputs(",", out);
    fputs("clusterCount", out);
    fputs(",", out);
    fputs("clusterRad", out);
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
    //ln
    fputs("\n", out);
}
void logResult(FILE *out, const DensityParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime)
{	//ModelArgs
    fprintf(out, "%i,%f,%u,%f,%llu,%llu,",
        modelArgs->agents,
        modelArgs->interactionRad,
        modelArgs->clusterCount,
        modelArgs->clusterRad,
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
    //ln
    fputs("\n", out);
    fflush(out);
}