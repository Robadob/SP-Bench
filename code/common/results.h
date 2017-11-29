#ifndef __results_h__
#define __results_h__
#include <ctime>
#include <memory>

struct Time_Init
{
	float overall;
	float initCurand;
	float kernel;
	float pbm;
	float freeCurand;
};
struct Time_Step
{
	float overall;
	float kernel;
	float texture;
};
struct Time_Step_dbl
{
	double overall;
	double kernel;
	double texture;
};
struct NeighbourhoodStats
{
    NeighbourhoodStats()
        :max(0), min(UINT_MAX), average(0.0f) { }
    unsigned int max;
    unsigned int min;
    float average;
};
enum ModelEnum : unsigned int
{
    Null = 0,
    Circles = 1,
    Density = 2,
};
struct ModelParams 
{
    ModelParams()
        : iterations(1000)
        , seed(12)
    {
        
    }
    //virtual ~ModelParams() = default;
    virtual const char *modelName() = 0;
    virtual const char *modelFlag() = 0;
    virtual ModelEnum enumerator() = 0;
    unsigned long long iterations;
    unsigned long long seed;
};

struct CirclesParams : ModelParams
{
    CirclesParams()
#if defined(_2D)
		: width(250)
#elif defined(_3D)
        : width(50)
#endif
		, density(0.01f)
		, interactionRad(5.0f)
		, attractionForce(0.5f)
		, repulsionForce(0.5f)
	{ }
	unsigned int width;
	float density;
	float interactionRad;
	float attractionForce;
	float repulsionForce;

    const char *modelName() override { return MODEL_NAME; };
    const char *modelFlag() override { return MODEL_FLAG; };
    ModelEnum enumerator() override { return ModelEnum::Circles; };
private:
    const char *MODEL_NAME = "Circles";
    const char *MODEL_FLAG = "-circles";
};

struct NullParams : ModelParams
{
    NullParams()
        : agents(16384)
        , density(0.125f)
        , interactionRad(5.0f)
    { }
    unsigned int agents;
    float density;
    float interactionRad;

    const char *modelName() override { return MODEL_NAME; };
    const char *modelFlag() override { return MODEL_FLAG; };
    ModelEnum enumerator() override { return ModelEnum::Null; };
private:
    const char *MODEL_NAME = "Null";
    const char *MODEL_FLAG = "-null";
};

struct DensityParams : ModelParams
{
    DensityParams()
        : agentsPerCluster(2048)
        , envWidth(50.0f)
        , interactionRad(5.0f)
        , clusterCount(5)
        , clusterRad(5.0f)
        , uniformDensity(0.0f)
    { }
    unsigned int agentsPerCluster;
    float envWidth;
    float interactionRad;
    unsigned int clusterCount;
    float clusterRad;
    float uniformDensity;

    const char *modelName() override { return MODEL_NAME; };
    const char *modelFlag() override { return MODEL_FLAG; };
    ModelEnum enumerator() override { return ModelEnum::Density; };
private:
    const char *MODEL_NAME = "Density";
    const char *MODEL_FLAG = "-density";
};
struct ArgData
{
    ArgData()
        : pipe(false)
        , profile(false)
        , device(0)
#ifdef _GL
        , GLwidth(1280)
        , GLheight(720)
#endif
        , model()
    {}
    bool pipe = false;
    bool profile = false;
    bool exportAgents = false;
    bool exportInit = false;
    unsigned int device;
#ifdef _GL
    unsigned int GLwidth;
    unsigned int GLheight;
#endif
    std::shared_ptr<ModelParams> model;
};
#endif