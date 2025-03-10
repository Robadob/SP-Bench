#include "near_neighbours/Neighbourhood.cuh"
#include <algorithm>
#include <string>
#include "cuda_profiler_api.h"
#ifdef _GL
#include "Visualisation/Visualisation.h"
#include "ParticleScene.h"
#endif
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
#include "results.h"
#include <fcntl.h>
#ifdef _MSC_VER
#include <io.h>
#endif
#include "export.h"
#include <memory>

ArgData parseArgs(int argc, char * argv[])
{
    unsigned long long seed = ULONG_MAX;
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
		//-profile (<uint>) (<uint>), Disables all console IO
		if (arg.compare("-profile") == 0)
		{
#ifdef _CUDACC_DEBUG_
            throw std::runtime_exception("'-profile' runtime arg is unavailable when code built with DEBUG flag.\n");
#endif
			data.profile = true;
            //Check if next arg is beginning of new arg
            if (i + 1<argc&&argv[i+1][0]!='-')
            {
                //Optional profiler start time
                data.prof_first = (unsigned int)strtoul(argv[++i], nullptr, 0)+1;//0-indexed->1-indexed
                if (i + 1<argc&&argv[i + 1][0] != '-')
                {
                    //Optional profiler end time
                    data.prof_last = (unsigned int)strtoul(argv[++i], nullptr, 0)+1;//0-indexed->1-indexed
                }
            }
		}
		//-seed <ulong>, Uses the specified rng seed, defaults to 12, 0 produces uniform initialisation
		else if (arg.compare("-seed") == 0)
		{
            seed = (unsigned int)strtoul(argv[++i], nullptr, 0);
		}
		//-device <uint>, Uses the specified cuda device, defaults to 0
		else if (arg.compare("-device") == 0)
		{
			data.device = (unsigned int)strtoul(argv[++i], nullptr, 0);
		}
#ifdef CIRCLES_MODEL
		//V2: -circles <uint> <float> <float> <ulong>
        //Enables circles model
	    //Sets the agents, density, forceModifier and iterations to be executed
		else if (arg.compare("-circles") == 0)
		{
            assert(!data.model);//Two model params passed at runtime?
            std::shared_ptr<CirclesParams> mdl = std::make_shared<CirclesParams>();
            //mdl->width = (unsigned int)strtoul(argv[++i], nullptr, 0);
            mdl->agents = (unsigned int)strtoul(argv[++i], nullptr, 0);
            mdl->density = (float)atof(argv[++i]);
            //mdl->interactionRad = (float)atof(argv[++i]);
            //mdl->attractionForce = (float)atof(argv[++i]);
            //mdl->repulsionForce = (float)atof(argv[++i]);
            mdl->forceModifier = (float)atof(argv[++i]);
            mdl->iterations = strtoul(argv[++i], nullptr, 0);
            data.model = mdl;
        }
#endif
#ifdef NULL_MODEL
        ////-null <uint> <float> <ulong>
        ////Enables null model
        ////Sets the agent count, density, interaction rad and iterations to be executed
        else if (arg.compare("-null") == 0)
        {
           assert(!data.model);//Two model params passed at runtime?
           std::shared_ptr<NullParams> mdl = std::make_shared<NullParams>();
           mdl->agents = (unsigned int)strtoul(argv[++i], nullptr, 0);
           mdl->density = (float)atof(argv[++i]);
           mdl->iterations = strtoul(argv[++i], nullptr, 0);
           data.model = mdl;
        }
        //-density <uint> <float> <float> <float> <uint> <float>
        //Enables null model with density initialisation
        //Sets the agentPerCluster, envWidth, clusterCount, clusterRad, interactionRad, uniformDensity and iterations
        else if (arg.compare("-density") == 0)
        {
            assert(!data.model);//Two model params passed at runtime?
            std::shared_ptr<DensityParams> mdl = std::make_shared<DensityParams>();
            mdl->agentsPerCluster = (unsigned int)strtoul(argv[++i], nullptr, 0);
            mdl->envWidth = (float)atof(argv[++i]);
            mdl->clusterCount = (unsigned int)strtoul(argv[++i], nullptr, 0);
            mdl->clusterRad = (float)atof(argv[++i]);
            mdl->interactionRad = (float)atof(argv[++i]);
            mdl->uniformDensity = (float)atof(argv[++i]);
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
                data.model->seed = 12;
                continue;
            }
#endif
#ifdef NULL_MODEL
            if (arg.compare("null") == 0)
            {
                data.model = std::make_shared<NullParams>();
                data.model->seed = 0;
                continue;
            }
            if (arg.compare("density") == 0)
            {
                auto a = std::make_shared<DensityParams>();
                data.model = a;
                a->uniformDensity = 0.1f;//Enough to show, but they wont interact
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
        else if (arg.compare("-steps") == 0)
        {
            data.exportSteps = true;
        }
	}
    //If seed is set, add to model data
    if (seed != ULONG_MAX &&data.model)
        data.model->seed = seed;
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
    float density = 1.0f;
#endif
    std::shared_ptr<CoreModel> model;
    switch (args.model->enumerator())
    {
#ifdef CIRCLES_MODEL
        case Circles:
        {
            std::shared_ptr<CirclesParams> _model = std::dynamic_pointer_cast<CirclesParams>(args.model);
            model = std::make_shared<CirclesModel>(_model->agents, _model->density, _model->forceModifier);
#ifdef _GL
            density = _model->density;
#endif
            break;
        }
#endif
#ifdef NULL_MODEL
        case Null:
        {
            std::shared_ptr<NullParams> _model = std::dynamic_pointer_cast<NullParams>(args.model);
            model = std::make_shared<NullModel>(_model->agents, _model->density);
#ifdef _GL
            density = _model->density;
#endif
            break;
        }
        case Density:
        {
            std::shared_ptr<DensityParams> _model = std::dynamic_pointer_cast<DensityParams>(args.model);
            model = std::make_shared<NullModel>(_model->envWidth, _model->interactionRad, _model->clusterCount, _model->agentsPerCluster, _model->uniformDensity);
            break;
        }
#endif
        default:
            assert(false);//Model not configured
    }
    //Arkward def, to ensure we keep initTimes const
    std::shared_ptr<DensityParams> _model = std::dynamic_pointer_cast<DensityParams>(args.model);
    //Need to init textures before creating the scene
    //Temp disable cluster init
    const Time_Init initTimes = _model
        ? model->initPopulationClusters(_model->clusterCount, _model->clusterRad, _model->agentsPerCluster, _model->seed)
        : model->initPopulation(args.model->seed);
    //const Time_Init initTimes = model->initPopulation(args.model->seed);
#ifdef _GL
    std::shared_ptr<ParticleScene> scene = std::make_shared<ParticleScene>(v, model, density);
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
    NeighbourhoodStats nhFirst, nhLast;
    Time_Step *stepsAll = nullptr;
    NeighbourhoodStats *nhAll = nullptr;
    if (args.exportSteps)
    {
        stepsAll = (Time_Step *)malloc(sizeof(Time_Step)* args.model->iterations);
        memset(stepsAll, 0, sizeof(Time_Step)* args.model->iterations);
        nhAll = (NeighbourhoodStats *)malloc(sizeof(NeighbourhoodStats)* args.model->iterations);
        memset(nhAll, 0, sizeof(NeighbourhoodStats)* args.model->iterations);
    }
    for (unsigned long long i = 1; i <= args.model->iterations; i++)
	{
        if (args.profile&&i == args.prof_first)
        {
            CUDA_CALL(cudaProfilerStart());
        }
        //Build array of times.
        if (args.exportSteps)
        {
            stepsAll[i-1] = model->step();
            nhAll[i - 1] = model->getPartition()->getNeighbourhoodStats();
            //Calculate averages
            average.overall += stepsAll[i - 1].overall / args.model->iterations;
            average.kernel += stepsAll[i - 1].kernel / args.model->iterations;
            average.texture += stepsAll[i - 1].texture / args.model->iterations;
        }
        else
        {
            const Time_Step iterTime = model->step();
            //Calculate averages
            average.overall += iterTime.overall / args.model->iterations;
            average.kernel += iterTime.kernel / args.model->iterations;
            average.texture += iterTime.texture / args.model->iterations;
            average.pbm.sort += iterTime.pbm.sort / args.model->iterations;
            average.pbm.reorder += iterTime.pbm.reorder / args.model->iterations;
            average.pbm.texcopy += iterTime.pbm.texcopy / args.model->iterations;
        }
#ifdef _GL
		//Pass count to visualisation
		scene->setCount(model->getPartition()->getLocationCount());
		v.render();
#endif
		if (!args.pipe&&!args.profile)
		{
			printf("\r%6llu/%llu", i, args.model->iterations);
		}
        if (i==1)
        {
            if (args.exportSteps)
            {
                nhFirst = nhAll[0];
            }
            else
            {
                nhFirst = model->getPartition()->getNeighbourhoodStats();
            }
        }
        if (args.profile&&i == args.prof_last)
        {
            CUDA_CALL(cudaProfilerStop());
        }
    }
    if (args.exportSteps)
    {
        nhLast = nhAll[args.model->iterations-1];
    }
    else
    {
        nhLast = model->getPartition()->getNeighbourhoodStats();
    }
	if (!args.pipe&&!args.profile)
	{
		printf("\nModel complete - Average Times\n");
		printf("Main kernel - %.3fs\n", average.kernel / 1000);
		printf("Build PBM   - %.3fs\n", average.texture / 1000);
        printf("Combined    - %.3fs\n", average.overall / 1000);
        printf("\nPBM Breakdown\n");
        printf("PBM Sort    - %.3fs\n", average.pbm.sort / 1000);
        printf("PBM Reorder - %.3fs\n", average.pbm.reorder / 1000);
        printf("PBM Texcopy - %.3fs\n", average.pbm.texcopy / 1000);
        printf("\n");
        printf("Neighbourhood stats\n");
        printf("First - Min:%d, Max:%d, Average:%f\n", nhFirst.min, nhFirst.max, nhFirst.average);
        printf("Last  - Min:%d, Max:%d, Average:%f\n", nhLast.min, nhLast.max, nhLast.average);
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
#ifdef _MSC_VER
        //Put stdout into binary mode (stops binary for '\n' being converted to '\r\n')
		setmode(fileno(stdout), O_BINARY);
#endif
        size_t modelSize = 0;
        switch (args.model->enumerator())
        {
#ifdef CIRCLES_MODEL
        case Circles:
            modelSize = sizeof(CirclesParams);
            break;
#endif
#ifdef NULL_MODEL
        case Null:
            modelSize = sizeof(NullParams);
            break;
        case Density:
            modelSize = sizeof(DensityParams);
            break;
#endif
        default:
            assert(false);//Model not configured
        }
        if (fwrite(args.model.get(), modelSize, 1, stdout) != 1)
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
        if (fwrite(&nhFirst, sizeof(NeighbourhoodStats), 1, stdout) != 1)
        {
            freopen("error.log", "a", stderr);
            fprintf(stderr, "Writing first NeighbourhoodStats failed.\n");
            fflush(stderr);
        };
        if (fwrite(&nhLast, sizeof(NeighbourhoodStats), 1, stdout) != 1)
        {
            freopen("error.log", "a", stderr);
            fprintf(stderr, "Writing last NeighbourhoodStats failed.\n");
            fflush(stderr);
        };
	}
	if (args.exportAgents)
    {
        if (!args.pipe&&!args.profile)
            printf("Exporting agents to '%s'...", "agents.txt");
        if (auto a = std::dynamic_pointer_cast<NullModel>(model))
        {
            exportNullAgents(model->getPartition(), "agents.txt", a->getResults());
        }
        else
        {
            exportAgents(model->getPartition(), "agents.txt");
        }
        if (!args.pipe&&!args.profile)
            printf("...Completed!\n");
    }
    if (args.exportSteps)
    {
        char filename[128];
        sprintf(filename, "steps-%s.csv", MOD_NAME_SHORT);
        if (!args.pipe)
            printf("Exporting steps to '%s'...", filename);
        exportSteps(argc, argv, stepsAll, nhAll, (unsigned int)args.model->iterations, filename);
        if (!args.pipe)
            printf("...Completed!\n");
        free(stepsAll);stepsAll=nullptr;
        free(nhAll);nhAll=nullptr;
    }
#ifdef _GL
    v.run();
    v.~Visualisation();
    scene.reset();
#endif
    //Shutdown model before CUDA device reset
    model.reset();

	cudaDeviceReset();
#ifdef _MSC_VER
    //Wait for input before exit on windows
	if (!args.pipe&&!args.profile)
	{
		getchar();
	}
#endif
    return 0;
}
