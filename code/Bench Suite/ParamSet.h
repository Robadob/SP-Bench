#ifndef __ParamSet_h__
#define __ParamSet_h__
#define _CRT_SECURE_NO_WARNINGS
#include <string>
#include <memory>
#include "results.h"
#include <vector>
struct Executable
{
	const char *name;
	const char *file;
};

struct ParamSet
{
	ParamSet()
		: start(nullptr), end1(nullptr), end2(nullptr), steps1(0), steps2(0) {}
	//Vars
	std::string name;
	std::shared_ptr<ModelParams> start;
	std::shared_ptr<ModelParams> end1;
	std::shared_ptr<ModelParams> end2;
	unsigned int steps1, steps2;
	//Methods
	bool validate() const;
	void execute() const;
	static void setBinDir(const char*d);
	static void setOutputDir(const char*d);
	static void setDeviceId(unsigned int d);
private:
	void run() const;
	void runCollated() const;
	void runCollated2D() const;
	static std::string BIN_DIR;
	static std::string OUT_DIR;
	static unsigned int DEVICE_ID;
	static std::vector<Executable> BINARIES;
	FILE *createLogFile(const char*timestamp) const;

	static void logCollatedHeader(FILE *out, const CirclesParams &modelArgs);
	static void logCollatedHeader(FILE *out, const NullParams &modelArgs);
    static void logCollatedHeader(FILE *out, const DensityParams &modelArgs);
    static void logCollatedHeader(FILE *out, const NetworkParams &modelArgs);
	static std::shared_ptr<ModelParams> interpolateParams(std::shared_ptr<ModelParams> start, std::shared_ptr<ModelParams> end, const unsigned int step, const unsigned int totalSteps);
	static std::shared_ptr<ModelParams> interpolateParams2D(std::shared_ptr<ModelParams> start, std::shared_ptr<ModelParams> end1, std::shared_ptr<ModelParams> end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2);

public:
	static void execString(const char* executable, CirclesParams modelArgs, char **rtn);
	static void execString(const char* executable, NullParams modelArgs, char **rtn);
    static void execString(const char* executable, DensityParams modelArgs, char **rtn);
    static void execString(const char* executable, NetworkParams modelArgs, char **rtn);
};

#endif //__ParamSet_h__