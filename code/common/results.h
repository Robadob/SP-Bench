#ifndef __results_h__
#define __results_h__
#include <ctime>
#include <climits>
#include <cstring>
#include <memory>

struct Time_Init
{
	float overall;
	float initCurand;
	float kernel;
	float pbm;
	float freeCurand;
};
struct PBM_Time
{
    float sort;
    float reorder;
    float texcopy;
};
struct Time_Step
{
	float overall;
	float kernel;
	float texture;
    PBM_Time pbm;
};
struct PBM_Time_dbl
{
    double sort;
    double reorder;
    double texcopy;
};
struct Time_Step_dbl
{
	double overall;
	double kernel;
    double texture;
    PBM_Time_dbl pbm;
};
struct NeighbourhoodStats
{
    NeighbourhoodStats()
        :max(0), min(UINT_MAX), average(0.0f) { }
    unsigned int max;
    unsigned int min;
    float average;
    float standardDeviation;
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
	virtual const char *modelName() const = 0;
	virtual const char *modelFlag() const = 0;
    virtual ModelEnum enumerator() const = 0;
    unsigned long long iterations;
    unsigned long long seed;
};

struct CirclesParams : ModelParams
{
    static CirclesParams makeEmpty()
    {
        CirclesParams a;
        memset(&a, 0, sizeof(CirclesParams));
        return a;
    }
    CirclesParams()
        : agents(16384)
        , density(0.01f)
		, forceModifier(0.5f)
	{ }

	unsigned int agents;
	float density;
    float forceModifier;

	const char *modelName( )const override { return MODEL_NAME; };
	const char *modelFlag() const override { return MODEL_FLAG; };
	ModelEnum enumerator() const override{ return ModelEnum::Circles; };
private:
    const char *MODEL_NAME = "Circles";
    const char *MODEL_FLAG = "-circles";
};

struct NullParams : ModelParams
{
    static NullParams makeEmpty()
    {
        NullParams a;
        memset(&a, 0, sizeof(NullParams));
        return a;
    }
    NullParams()
        : agents(16384)
        , density(1.5f)
    { }
    unsigned int agents;
    float density;

	const char *modelName() const override { return MODEL_NAME; };
	const char *modelFlag() const override { return MODEL_FLAG; };
	ModelEnum enumerator() const override { return ModelEnum::Null; };
private:
    const char *MODEL_NAME = "Null";
    const char *MODEL_FLAG = "-null";
};

struct DensityParams : ModelParams
{
    static DensityParams makeEmpty()
    {
        DensityParams a;
        memset(&a, 0, sizeof(DensityParams));
        return a;
    }
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

	const char *modelName() const override { return MODEL_NAME; };
	const char *modelFlag() const override { return MODEL_FLAG; };
	ModelEnum enumerator() const override { return ModelEnum::Density; };
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
    unsigned int prof_first = 0;
    unsigned int prof_last = UINT_MAX;
    bool exportAgents = false;
    bool exportInit = false;
    bool exportSteps = false;
    unsigned int device;
#ifdef _GL
    unsigned int GLwidth;
    unsigned int GLheight;
#endif
    std::shared_ptr<ModelParams> model;
};
#endif