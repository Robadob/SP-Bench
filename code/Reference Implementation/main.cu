#include "Neighbourhood.cuh"
#include "Circles.cuh"
#include <algorithm>
#include <string>
#ifdef _GL
#include "Visualisation/Visualisation.h"
#include "ParticleScene.h"
#endif
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
#include "results.h"
#include <fcntl.h>
#include <io.h>
struct ArgData
{
	ArgData()
		: pipe(false)
		, device(0)
		, GLwidth(1280)
		, GLheight(720)
		, model()
	{}
	bool pipe = false;
	unsigned int device;
#ifdef _GL
	unsigned int GLwidth;
	unsigned int GLheight;
#endif
	ModelParams model;
};
ArgData parseArgs(int argc, char * argv[])
{
	//Init so defaults are used
	ArgData data = { };
	for (int i = 0; i < argc; i++)
	{
		std::string arg(argv[i]);
		std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);
		//-pipe, Runs in piped mode where results are written as binary to a pipe
		if (arg.compare("-pipe") == 0)
		{
			data.pipe = true;
		}
		//-device <uint>, Uses the specified cuda device, defaults to 0
		else if (arg.compare("-device") == 0)
		{
			data.device = (unsigned int)strtoul(argv[++i], nullptr, 0);
		}
		//-model <uint> <float> <float> <float> <float> <ulong>, Sets the width, density, interaction rad, attractionForce, repulsionForce and iterations to be executed
		else if (arg.compare("-model") == 0)
		{
			data.model.width = (unsigned int)strtoul(argv[++i],nullptr,0);
			data.model.density = (float)atof(argv[++i]);
			data.model.interactionRad = (float)atof(argv[++i]);
			data.model.attractionForce = (float)atof(argv[++i]);
			data.model.repulsionForce = (float)atof(argv[++i]);
			data.model.iterations = strtoul(argv[++i], nullptr, 0);
		}
#ifdef _GL
		//-resolution <uint> <uint>, Sets the width and height of the GL window
		else if (arg.compare("-resolution") == 0 || arg.compare("-res") == 0)
		{
			data.GLwidth = (unsigned int)strtoul(argv[++i], nullptr, 0);
			data.GLheight = (unsigned int)strtoul(argv[++i], nullptr, 0);
		}
#endif
	}
	return data;
}
int main(int argc, char * argv[])
{
	ArgData args = parseArgs(argc, argv);
	cudaSetDevice(args.device);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

#ifdef _GL
	Visualisation v("Visulisation Example", args.GLwidth, args.GLheight);//Need to init GL before creating CUDA textures
#endif
	Circles<SpatialPartition> model(args.model.width, args.model.density, args.model.interactionRad*2, args.model.attractionForce, args.model.repulsionForce);
	const Time_Init initTimes = model.initPopulation();//Need to init textures before creating the scene
#ifdef _GL
	ParticleScene<SpatialPartition> *scene = new ParticleScene<SpatialPartition>(v, model);
#endif

	//Init model
	if (!args.pipe)
	{
		printf("Init Complete - Times\n");
		printf("CuRand init - %.3fs\n", initTimes.initCurand / 1000);
		printf("Main kernel - %.3fs\n", initTimes.kernel / 1000);
		printf("Build PBM   - %.3fs\n", initTimes.pbm / 1000);
		printf("CuRand free - %.3fs\n", initTimes.freeCurand / 1000);
		printf("Combined    - %.3fs\n", initTimes.overall / 1000);
		printf("\n");
	}
	//Start visualisation
	//v.runAsync();
	//v.run();
	//Do iterations
	Time_Step_dbl average = {};//init
	for (unsigned long long i = 1; i <= args.model.iterations; i++)
	{
		const Time_Step iterTime = model.step();
		//Calculate averages
		average.overall += iterTime.overall / args.model.iterations;
		average.kernel += iterTime.kernel / args.model.iterations;
		average.texture += iterTime.texture / args.model.iterations;
#ifdef _GL
		//Pass count to visualisation
		scene->setCount(model.getPartition()->getLocationCount());
		v.render();
#endif
		if (!args.pipe)
		{
			printf("\r%6llu/%llu", i, args.model.iterations);
		}
	}
	if (!args.pipe)
	{
		printf("\nModel complete - Average Times\n");
		printf("Main kernel - %.3fs\n", average.kernel / 1000);
		printf("Build PBM   - %.3fs\n", average.texture / 1000);
		printf("Combined    - %.3fs\n", average.overall / 1000);
		printf("\n");
	}

    //Calculate final timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float totalTime;
    cudaEventElapsedTime(&totalTime, start, stop);

	if (!args.pipe)
	{
		printf("Total Runtime: %.3fs\n", totalTime / 1000);
	}
	//Print piped data
	if (args.pipe)
	{
		//FILE *pipe = fopen("CON", "wb+");// _popen("", "wb");
		setmode(fileno(stdout), O_BINARY);
		fwrite(&initTimes, sizeof(Time_Init), 1, stdout);
		fwrite(&average, sizeof(Time_Step_dbl), 1, stdout);
		fwrite(&totalTime, sizeof(float), 1, stdout);
		//_pclose(pipe);
	}
#ifdef _GL
    v.run();
#endif

    //Wait for input before exit
	if (!args.pipe)
	{
		getchar();
	}
    return 0;
}
