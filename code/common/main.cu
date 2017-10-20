#include "near_neighbours/Neighbourhood.cuh"
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
#include <memory>

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
		//-seed <ulong>, Uses the specified rng seed, defaults to 12, 0 produces uniform initialisation
		else if (arg.compare("-seed") == 0)
		{
			data.seed = (unsigned int)strtoul(argv[++i], nullptr, 0);
		}
		//-device <uint>, Uses the specified cuda device, defaults to 0
		else if (arg.compare("-device") == 0)
		{
			data.device = (unsigned int)strtoul(argv[++i], nullptr, 0);
		}
#ifdef CIRCLES_MODEL
		//-circles <uint> <float> <float> <float> <float> <ulong>
        //Enables circles model
	    //Sets the width, density, interaction rad, attractionForce, repulsionForce and iterations to be executed
		else if (arg.compare("-circles") == 0)
		{
            assert(!data.model);//Two model params passed at runtime?
            std::shared_ptr<CirclesParams> mdl = std::make_shared<CirclesParams>();
            mdl->width = (unsigned int)strtoul(argv[++i], nullptr, 0);
            mdl->density = (float)atof(argv[++i]);
            mdl->interactionRad = (float)atof(argv[++i]);
            mdl->attractionForce = (float)atof(argv[++i]);
            mdl->repulsionForce = (float)atof(argv[++i]);
            mdl->iterations = strtoul(argv[++i], nullptr, 0);
            data.model = mdl;
        }
#endif
#ifdef NULL_MODEL
        ////-null <uint> <float> <float> <ulong>
        ////Enables null model
        ////Sets the agent count, density, interaction rad and iterations to be executed
        else if (arg.compare("-null") == 0)
        {
           assert(!data.model);//Two model params passed at runtime?
           std::shared_ptr<NullParams> mdl = std::make_shared<NullParams>();
           mdl->agents = (unsigned int)strtoul(argv[++i], nullptr, 0);
           mdl->density = (float)atof(argv[++i]);
           mdl->interactionRad = (float)atof(argv[++i]);
           mdl->iterations = strtoul(argv[++i], nullptr, 0);
           data.model = mdl;
        }
        //-density <uint> <float> <float> <uint> <float>
        //Enables null model with density initialisation
        //Sets the agentCount, envWidth, interactionRad, clusterCount, clusterRad and iterations
        else if (arg.compare("-density") == 0)
        {
            assert(!data.model);//Two model params passed at runtime?
            std::shared_ptr<DensityParams> mdl = std::make_shared<DensityParams>();
            mdl->agents = (unsigned int)strtoul(argv[++i], nullptr, 0);
            mdl->envWidth = (float)atof(argv[++i]);
            mdl->clusterCount = (unsigned int)strtoul(argv[++i], nullptr, 0);
            mdl->clusterRad = (float)atof(argv[++i]);
            mdl->iterations = strtoul(argv[++i], nullptr, 0);
            data.model = mdl;
        }
#endif
        else if (arg.compare("-demo") == 0)
        {
            assert(!data.model);//Two model params passed at runtime?
            arg = argv[++i];
#ifdef CIRCLES_MODEL
            if(arg.compare("circles") == 0)
            {
                data.model = std::make_shared<CirclesParams>();
                data.seed = 12;
                continue;
            }
#endif
#ifdef NULL_MODEL
            if (arg.compare("null") == 0)
            {
                data.model = std::make_shared<NullParams>();
                data.seed = 12;
                continue;
            }
            if (arg.compare("density") == 0)
            {
                data.model = std::make_shared<DensityParams>();
                continue;
            }
#endif
        }
#ifdef _GL
		//-resolution <uint> <uint>, Sets the width and height of the GL window
		else if (arg.compare("-resolution") == 0 || arg.compare("-res") == 0)
		{
			data.GLwidth = (unsigned int)strtoul(argv[++i], nullptr, 0);
			data.GLheight = (unsigned int)strtoul(argv[++i], nullptr, 0);
		}
#endif
		else if (arg.compare("-export")==0)
		{
			data.exportAgents = true;
		}
		else if (arg.compare("-init") == 0)
		{
			data.exportInit = true;
		}
	}
	return data;
}
int main(int argc, char * argv[])
{
	ArgData args = parseArgs(argc, argv);
    assert(args.model);//No model selected!
	cudaError_t status = cudaSetDevice(args.device);
	// If there were no errors, proceed.
	if (status == cudaSuccess){
		if (!args.pipe)
		{
			// Get properties
			cudaDeviceProp props;
			status = cudaGetDeviceProperties(&props, args.device);
			// If we have properties, print the device.
			if (status == cudaSuccess){
				fprintf(stdout, "Device: %s\n  pci %d bus %d\n  tcc %d\n  SM %d%d\n\n", props.name, props.pciDeviceID, props.pciBusID, props.tccDriver, props.major, props.minor);
			}
		}
	}
	else {
		fprintf(stderr, "Error setting CUDA Device %d.\n", args.device);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

    if (!args.pipe&&!args.profile)
    {
        printf("Active Mod: %s\n", MOD_NAME);
    }

    //Begin instrumentation
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

#ifdef _GL
	Visualisation v("Visulisation Example", args.GLwidth, args.GLheight);//Need to init GL before creating CUDA textures
#endif
    std::shared_ptr<CoreModel> model;
    
    switch (args.model->enumerator())
    {
#ifdef CIRCLES_MODEL
        case Circles:
        {
            std::shared_ptr<CirclesParams> _model = std::dynamic_pointer_cast<CirclesParams>(args.model);
            model = std::make_shared<CirclesModel>(_model->width, _model->density, _model->interactionRad * 2, _model->attractionForce, _model->repulsionForce);
            break;
        }
#endif
#ifdef NULL_MODEL
        case Null:
        {
            std::shared_ptr<NullParams> _model = std::dynamic_pointer_cast<NullParams>(args.model);
            model = std::make_shared<NullModel>(_model->agents, _model->density, _model->interactionRad);
            break;
        }
        case Density:
        {
            std::shared_ptr<DensityParams> _model = std::dynamic_pointer_cast<DensityParams>(args.model);
            model = std::make_shared<NullModel>(_model->envWidth, _model->interactionRad, _model->agents);
            break;
        }
#endif
        default:
            assert(false);//Model not configured
    }
    //Arkward def, to ensure we keep initTimes const
    std::shared_ptr<DensityParams> _model = std::dynamic_pointer_cast<DensityParams>(args.model);
    //Need to init textures before creating the scene
    const Time_Init initTimes = args.model->enumerator() == Density
        ? model->initPopulationClusters(_model->clusterCount, _model->clusterRad, args.seed)
        : model->initPopulation(args.seed);
#ifdef _GL
	std::shared_ptr<ParticleScene> scene = std::make_shared<ParticleScene>(v, model);
#endif

	//Init model
	if (!args.pipe&&!args.profile)
	{
		printf("Agents: %d\n", model->getPartition()->getLocationCount());
		printf("Init Complete - Times\n");
		printf("CuRand init - %.3fs\n", initTimes.initCurand / 1000);
		printf("Main kernel - %.3fs\n", initTimes.kernel / 1000);
		printf("Build PBM   - %.3fs\n", initTimes.pbm / 1000);
		printf("CuRand free - %.3fs\n", initTimes.freeCurand / 1000);
		printf("Combined    - %.3fs\n", initTimes.overall / 1000);
		printf("\n");
	}
	if (args.exportInit)
	{
		exportPopulation(model->getPartition(), args, "init.xml");
	}
	//Start visualisation
	//v.runAsync();
	//v.run();
	//Do iterations
	Time_Step_dbl average = {};//init
    for (unsigned long long i = 1; i <= args.model->iterations; i++)
	{
		const Time_Step iterTime = model->step();
		//Calculate averages
        average.overall += iterTime.overall / args.model->iterations;
        average.kernel += iterTime.kernel / args.model->iterations;
        average.texture += iterTime.texture / args.model->iterations;
#ifdef _GL
		//Pass count to visualisation
		scene->setCount(model->getPartition()->getLocationCount());
		v.render();
#endif
		if (!args.pipe&&!args.profile)
		{
			printf("\r%6llu/%llu", i, args.model->iterations);
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
        if (fwrite(&args.model, sizeof(CirclesParams), 1, stdout) != 1)
		{
			freopen("error.log", "a", stderr);
			fprintf(stderr, "Writing model params failed.\n"); 
			fflush(stderr);
		};
		if (fwrite(&model->agentMax, sizeof(unsigned int), 1, stdout) != 1)
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
	if (args.exportAgents)
	{
		exportAgents(model->getPartition(), "agents.txt");
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
