#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <memory>
#include <string>

#include "results.h"
#include <ctime>

void execString(const char* executable, ModelParams model, char **rtn);
bool executeBenchmark(const char* executable, ModelParams modelArgs, ModelParams *modelparamOut, unsigned int *agentCount, Time_Init *initRes, Time_Step_dbl *stepRes, float *totalTime);
void logResult(FILE *out, const ModelParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime);
void logHeader(FILE *out);
const char *EXE_REFERENCE = "Release-Reference Implementation.exe";
const char *DIR_X64 = "..\\bin\\x64\\";
int main(int argc, char* argv[])
{
	//Create log
	std::string logPath("out-");
	logPath = logPath.append(std::to_string(time(0)));
	logPath = logPath.append(".csv");
	FILE *log = fopen(logPath.c_str(), "w");
	if (!log)
		return 1;
	//Problem Scale
	//Init model arg start
	ModelParams start = {};
	start.iterations = 1000;
	start.density = 0.01f;
	start.interactionRad = 5.0f;
	start.width = 50;
	//Init model arg end
	ModelParams end = {};
	end.iterations = 1000;
	end.density = 0.01f;
	end.interactionRad = 5.0f;
	end.width = 300;	
	//Init step count
	const int steps = 26;

	//Neighbourhood Scale
	////Init model arg start
	//ModelParams start = {};
	//start.iterations = 1000;
	//start.density = 0.01f;
	//start.interactionRad = 1.0f;
	//start.width = 100;
	////Init model arg end
	//ModelParams end = {};
	//end.iterations = 1000;
	//end.density = 0.01f;
	//end.interactionRad = 15.0f;
	//end.width = 100;
	////Init step count
	//const int steps = 16;
	const float stepsM1 =(float) steps-1.0f;
	//Create log header
	logHeader(log);
	//Create objects for use within the loop
	Time_Init initRes;
	Time_Step_dbl stepRes;
	ModelParams modelArgs;
	ModelParams modelParamsOut;
	unsigned int agentCount;
	float totalTime;
	//For each benchmark
	for (unsigned int i = 0; i < steps;i++)
	{
		printf("\rExecuting run %i/%i",i,(int)stepsM1);
		//Interpolate model
		modelArgs.width = start.width + (unsigned int)((i / stepsM1)*(end.width - start.width));
		modelArgs.density = start.density + ((i / stepsM1)*(end.density - start.density));
		modelArgs.interactionRad = start.interactionRad + ((i / stepsM1)*(end.interactionRad - start.interactionRad));
		modelArgs.attractionForce = start.attractionForce + ((i / stepsM1)*(end.attractionForce - start.attractionForce));
		modelArgs.repulsionForce = start.repulsionForce + ((i / stepsM1)*(end.repulsionForce - start.repulsionForce));
		modelArgs.iterations = start.iterations + (unsigned long long)((i / stepsM1)*(end.iterations - start.iterations));
		//Clear output structures
		memset(&modelParamsOut, 0, sizeof(ModelParams));
		memset(&stepRes, 0, sizeof(Time_Step_dbl));
		memset(&initRes, 0, sizeof(Time_Init));
		agentCount = 0;
		totalTime = 0;
		//executeBenchmark
		if (!executeBenchmark(EXE_REFERENCE, modelArgs, &modelParamsOut, &agentCount, &initRes, &stepRes, &totalTime))
			return 1;
		//logResult
		logResult(log, &modelParamsOut, agentCount, &initRes, &stepRes, totalTime);
	}
	//Close log
	fclose(log);
	printf("\nComplete\n");
}

bool executeBenchmark(const char* executable, ModelParams modelArgs, ModelParams *modelparamOut, unsigned int *agentCount, Time_Init *initRes, Time_Step_dbl *stepRes, float *totalTime)
{
	char *command;
	bool rtn = true;
	execString(executable, modelArgs, &command);
	std::shared_ptr<FILE> pipe(_popen(command, "rb"), _pclose);
	if (!pipe.get()) return false;
	if (fread(modelparamOut, sizeof(ModelParams), 1, pipe.get()) != 1)
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

void execString(const char* executable, ModelParams modelArgs, char **rtn)
{
	std::string buffer("\"");
	buffer = buffer.append(DIR_X64);
	buffer = buffer.append(executable);
	buffer = buffer.append("\"");
	buffer = buffer.append(" -pipe");
	buffer = buffer.append(" ");
	buffer = buffer.append(" -device");
	buffer = buffer.append(" ");
	buffer = buffer.append(std::to_string(0));
	buffer = buffer.append(" ");
	buffer = buffer.append(" -model");
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
	const char *src = buffer.c_str();
	*rtn = (char *)malloc(sizeof(char*)*(buffer.length() + 1));
	memcpy(*rtn, src, sizeof(char*)*(buffer.length() + 1));
}
void logHeader(FILE *out)
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
	fputs("agents", out);
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
void logResult(FILE *out, const ModelParams* modelArgs, const unsigned int agentCount, const Time_Init *initRes, const Time_Step_dbl *stepRes, const float totalTime)
{	//ModelArgs
	fprintf(out, "%i,%f,%f,%f,%f,%llu,",
		modelArgs->width,
		modelArgs->density,
		modelArgs->interactionRad,
		modelArgs->attractionForce,
		modelArgs->repulsionForce,
		modelArgs->iterations
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