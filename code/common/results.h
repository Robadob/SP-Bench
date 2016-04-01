#ifndef __results_h__
#define __results_h__

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
		: width(50)
		, density(0.005f)
		, interactionRad(10.0f)
		, attractionForce(0.0001f)
		, repulsionForce(0.0001f)
		, iterations(10000)
	{ }
	unsigned int width;
	float density;
	float interactionRad;
	float attractionForce;
	float repulsionForce;
	unsigned long long iterations;
};
#endif