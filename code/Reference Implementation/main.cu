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
#include "export.h"

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
		//-profile, Disables all console IO
		if (arg.compare("-profile") == 0)
		{
			data.profile = true;
		}
		//-seed <ulong>, Uses the specified rng seed, defaults to 12
		else if (arg.compare("-seed") == 0)
		{
			data.model.seed = (unsigned int)strtoul(argv[++i], nullptr, 0);
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
	const Time_Init initTimes = model.initPopulation(args.model.seed);//Need to init textures before creating the scene
#ifdef _GL
	ParticleScene<SpatialPartition> *scene = new ParticleScene<SpatialPartition>(v, model);
#endif
	exportPopulation(model.getPartition(), &args.model, "text.xml");
	//Init model
	if (!args.pipe&&!args.profile)
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
		if (!args.pipe&&!args.profile)
		{
			printf("\r%6llu/%llu", i, args.model.iterations);
		}
	}
	if (!args.pipe&&!args.profile)
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

	if (!args.pipe&&!args.profile)
	{
		printf("Total Runtime: %.3fs\n", totalTime / 1000);
	}
	//Print piped data
	if (args.pipe)
	{
		//FILE *pipe = fopen("CON", "wb+");// _popen("", "wb");
		setmode(fileno(stdout), O_BINARY);
		if (fwrite(&args.model, sizeof(ModelParams), 1, stdout) != 1)
		{
			freopen("error.log", "a", stderr);
			fprintf(stderr, "Writing model params failed.\n"); 
			fflush(stderr);
		};
		if (fwrite(&model.agentMax, sizeof(unsigned int), 1, stdout) != 1)
		{

			freopen("error.log", "a", stderr);
			fprintf(stderr, "Writing agent max failed.\n"); 
			fflush(stderr);
		};
		if (fwrite(&initTimes, sizeof(Time_Init), 1, stdout) != 1)
		{
			freopen("error.log", "a", stderr);
			fprintf(stderr, "Writing init times failed.\n"); 
			fflush(stderr);
		};
		if (fwrite(&average, sizeof(Time_Step_dbl), 1, stdout) != 1)
		{
			freopen("error.log", "a", stderr);
			fprintf(stderr, "Writing step times failed.\n");
			fflush(stderr);
		};
		if (fwrite(&totalTime, sizeof(float), 1, stdout) != 1)
		{
			freopen("error.log", "a", stderr);
			fprintf(stderr, "Writing total time failed.\n"); 
			fflush(stderr);
		};
	}
#ifdef _GL
    v.run();
#endif
	cudaDeviceReset();
    //Wait for input before exit
	if (!args.pipe&&!args.profile)
	{
		getchar();
	}
    return 0;
}
