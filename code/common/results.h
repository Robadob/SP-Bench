#ifndef __results_h__
#define __results_h__
#include <ctime>

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

struct ModelParams
{
	ModelParams()
		: width(100)
		, density(0.005f)
		, interactionRad(10.0f)
		, attractionForce(0.00001f)
		, repulsionForce(0.00001f)
		, iterations(5000)
		, seed(12)
	{ }
	unsigned int width;
	float density;
	float interactionRad;
	float attractionForce;
	float repulsionForce;
	unsigned long long iterations;
	unsigned long long seed;
};
#endif