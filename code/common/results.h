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
enum ModelEnum : unsigned int
{
    Null = 0,
    Circles = 1,
    Density = 2,
};
struct ModelParams 
{
    ModelParams()
        : iterations(5000)
    {
        
    }
    //virtual ~ModelParams() = default;
    virtual const char *modelName() = 0;
    virtual const char *modelFlag() = 0;
    virtual ModelEnum enumerator() = 0;
    unsigned long long iterations;
};

struct CirclesParams : ModelParams
{
    CirclesParams()
		: width(100)
		, density(0.005f)
		, interactionRad(10.0f)
		, attractionForce(0.00001f)
		, repulsionForce(0.00001f)
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
class SpatialPartition;
class Model
{
public:
    Model(const unsigned int agentMax)
        :agentMax(agentMax)
    { }
    virtual ~Model() = default;
    virtual const Time_Init initPopulation(const unsigned long long rngSeed = 12)=0;
    virtual const Time_Step step()=0;
    virtual std::shared_ptr<SpatialPartition> getPartition() = 0;
    const unsigned int agentMax;
};


struct ArgData
{
    ArgData()
        : pipe(false)
        , profile(false)
        , device(0)
        , seed(12)
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
    unsigned long long seed;
#ifdef _GL
    unsigned int GLwidth;
    unsigned int GLheight;
#endif
    std::shared_ptr<ModelParams> model;
};
#endif