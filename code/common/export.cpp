#include "export.h"
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <float.h>
//When building rapid xml headers with CUDA, we get this warning (doesn't occur with regular compiler)
//http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
//-Xcudafe "--diag_suppress=expr_not_integral_constant"
#include "rapidxml/rapidxml.hpp"
#include "rapidxml/rapidxml_print.hpp"

void exportPopulation(std::shared_ptr<SpatialPartition> s, const ArgData &args, const char *path)
{
	unsigned int count = s->getLocationCount();
	std::string outFile = std::string(path);
	if (outFile.empty())
	{
		printf("Err: Output file name must be specified\n");
		exit(1);
	}
	//Actually do the generation
	rapidxml::xml_document<char> doc;
	//Common strings
	char *states_node_str = doc.allocate_string("states");
	char *itno_node_str = doc.allocate_string("itno");
	char *xagent_node_str = doc.allocate_string("xagent");
	char *name_node_str = doc.allocate_string("name");
	char *id_node_str = doc.allocate_string("id");
	char *x_node_str = doc.allocate_string("x");
	char *y_node_str = doc.allocate_string("y");
	char *z_node_str = doc.allocate_string("z");
	char *fx_node_str = doc.allocate_string("fx");
	char *fy_node_str = doc.allocate_string("fy");
	char *fz_node_str = doc.allocate_string("fz");
	char *circle_str = doc.allocate_string("Circle");
	char *zero_pt_zero_str = doc.allocate_string("0.0");
	char *params_node_str = doc.allocate_string("constants");
	char *width_node_str = doc.allocate_string("width");
	char *rad_node_str = doc.allocate_string("interaction-radius");
	char *density_node_str = doc.allocate_string("density");
	char *seed_node_str = doc.allocate_string("seed");
	char *attract_node_str = doc.allocate_string("attraction-force");
	char *repel_node_str = doc.allocate_string("repulsion-force");
	//temp stuff
	char buffer[1024];
	float *d_bufferPtr;
	LocationMessages *d_lm = s->d_getLocationMessages();
	LocationMessages lm;
#ifdef AOS_MESSAGES
    lm.location = (DIMENSIONS_VEC*)malloc(sizeof(DIMENSIONS_VEC)*count);
    CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_lm->location, sizeof(DIMENSIONS_VEC*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(lm.location, d_bufferPtr, sizeof(DIMENSIONS_VEC)*count, cudaMemcpyDeviceToHost));
#else
	lm.locationX = (float*)malloc(sizeof(float)*count);
	CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_lm->locationX, sizeof(float*), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(lm.locationX, d_bufferPtr, sizeof(float)*count, cudaMemcpyDeviceToHost));
	lm.locationY = (float*)malloc(sizeof(float)*count);
	CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_lm->locationY, sizeof(float*), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(lm.locationY, d_bufferPtr, sizeof(float)*count, cudaMemcpyDeviceToHost));
#ifdef _3D
	lm.locationZ = (float*)malloc(sizeof(float)*count);
	CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_lm->locationZ, sizeof(float*), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(lm.locationZ, d_bufferPtr, sizeof(float)*count, cudaMemcpyDeviceToHost));
#endif
#endif
	//states
	rapidxml::xml_node<> *states_node = doc.allocate_node(rapidxml::node_element, states_node_str);
	doc.append_node(states_node);
	//itno
	rapidxml::xml_node<> *itno_node = doc.allocate_node(rapidxml::node_element, itno_node_str, doc.allocate_string("0"));
	states_node->append_node(itno_node);
	//model params
	rapidxml::xml_node<> *params_node = doc.allocate_node(rapidxml::node_element, params_node_str);
    {
        //Model name
        sprintf(buffer, "%s", args.model->modelName());
        rapidxml::xml_node<> *name_node = doc.allocate_node(rapidxml::node_element, name_node_str, doc.allocate_string(buffer));
        params_node->append_node(name_node);
        //seed
        sprintf(buffer, "%llu", args.model->seed);
        rapidxml::xml_node<> *seed_node = doc.allocate_node(rapidxml::node_element, seed_node_str, doc.allocate_string(buffer));
        params_node->append_node(seed_node);
        //Model specific params
        if (args.model->enumerator() == ModelEnum::Null)
        {
            assert(false);//Not yet implemented
        }
        else if (args.model->enumerator() == ModelEnum::Circles)
        {
            std::shared_ptr<CirclesParams> _model = std::dynamic_pointer_cast<CirclesParams>(args.model);
		    ////width
            //sprintf(buffer, "%d", _model->width);
		    //rapidxml::xml_node<> *width_node = doc.allocate_node(rapidxml::node_element, width_node_str, doc.allocate_string(buffer));
		    //params_node->append_node(width_node);	
            //agents
            sprintf(buffer, "%d", _model->agents);
		    rapidxml::xml_node<> *width_node = doc.allocate_node(rapidxml::node_element, width_node_str, doc.allocate_string(buffer));
		    params_node->append_node(width_node);
      //      //interaction rad
      //      sprintf(buffer, "%f", _model->interactionRad);
		    //rapidxml::xml_node<> *rad_node = doc.allocate_node(rapidxml::node_element, rad_node_str, doc.allocate_string(buffer));
		    //params_node->append_node(rad_node);
		    //density
            sprintf(buffer, "%f", _model->density);
		    rapidxml::xml_node<> *density_node = doc.allocate_node(rapidxml::node_element, density_node_str, doc.allocate_string(buffer));
		    params_node->append_node(density_node);
		    ////att force
            //sprintf(buffer, "%f", _model->attractionForce);
		    //rapidxml::xml_node<> *attract_node = doc.allocate_node(rapidxml::node_element, attract_node_str, doc.allocate_string(buffer));
		    //params_node->append_node(attract_node);
		    ////rep force
            //sprintf(buffer, "%f", _model->repulsionForce);
		    //rapidxml::xml_node<> *repel_node = doc.allocate_node(rapidxml::node_element, repel_node_str, doc.allocate_string(buffer));
            //params_node->append_node(repel_node);
            //forceModifier force
            sprintf(buffer, "%f", _model->forceModifier);
            rapidxml::xml_node<> *repel_node = doc.allocate_node(rapidxml::node_element, repel_node_str, doc.allocate_string(buffer));
            params_node->append_node(repel_node);
        }
        else if (args.model->enumerator() == ModelEnum::Density)
        {
            assert(false);//Not yet implemented
        }
        else
        {
            assert(false);
        }
	}
	states_node->append_node(params_node);

	//xagent each
	for (unsigned int i = 0; i < count; i++)
	{
		rapidxml::xml_node<> *xagent_node = doc.allocate_node(rapidxml::node_element, xagent_node_str);
		{
			rapidxml::xml_node<> *name_node = doc.allocate_node(rapidxml::node_element, name_node_str, circle_str);
			xagent_node->append_node(name_node);

			sprintf(buffer, "%d", i);
			rapidxml::xml_node<> *id_node = doc.allocate_node(rapidxml::node_element, id_node_str, doc.allocate_string(buffer));
			xagent_node->append_node(id_node);

#ifdef AOS_MESSAGES
            sprintf(buffer, "%.*g", 9, lm.location[i].x);
#else
			sprintf(buffer, "%.*g", 9, lm.locationX[i]);
#endif
			rapidxml::xml_node<> *x_node = doc.allocate_node(rapidxml::node_element, x_node_str, doc.allocate_string(buffer));
			xagent_node->append_node(x_node);

#ifdef AOS_MESSAGES
            sprintf(buffer, "%.*g", 9, lm.location[i].y);
#else
			sprintf(buffer, "%.*g", 9, lm.locationY[i]);
#endif
			rapidxml::xml_node<> *y_node = doc.allocate_node(rapidxml::node_element, y_node_str, doc.allocate_string(buffer));
			xagent_node->append_node(y_node);
#ifdef _3D
#ifdef AOS_MESSAGES
            sprintf(buffer, "%.*g", 9, lm.location[i].z);
#else
			sprintf(buffer, "%.*g", 9, lm.locationZ[i]);
#endif
			rapidxml::xml_node<> *z_node = doc.allocate_node(rapidxml::node_element, z_node_str, doc.allocate_string(buffer));
			xagent_node->append_node(z_node);
#endif
            //Skip printing velocity defaults
			//rapidxml::xml_node<> *fx_node = doc.allocate_node(rapidxml::node_element, fx_node_str, zero_pt_zero_str);
			//xagent_node->append_node(fx_node);

			//rapidxml::xml_node<> *fy_node = doc.allocate_node(rapidxml::node_element, fy_node_str, zero_pt_zero_str);
			//xagent_node->append_node(fy_node);

			//rapidxml::xml_node<> *fz_node = doc.allocate_node(rapidxml::node_element, fz_node_str, zero_pt_zero_str);
			//xagent_node->append_node(fz_node);
		}
		states_node->append_node(xagent_node);
	}

	//Actually do the output
	std::ofstream f;
	f.open(outFile);

	f << doc;

	f.close();
#ifdef AOS_MESSAGES
    free(lm.location);
#else
	free(lm.locationX);
    free(lm.locationY);
#ifdef _3D
	free(lm.locationZ);
#endif
#endif
}

void exportAgents(std::shared_ptr<SpatialPartition> s, const char *path)
{
	
	int len = s->getLocationCount();
	std::ofstream oFile;
	oFile.open(path);
	char buffer[1024];
	if (oFile.is_open())
	{
		//allocate
		float *d_bufferPtr;
		LocationMessages *d_lm = s->d_getLocationMessages();
        LocationMessages lm;
#ifdef AOS_MESSAGES
        lm.location = (DIMENSIONS_VEC*)malloc(sizeof(DIMENSIONS_VEC)*len);
        CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_lm->location, sizeof(DIMENSIONS_VEC*), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(lm.location, d_bufferPtr, sizeof(DIMENSIONS_VEC)*len, cudaMemcpyDeviceToHost));
#else
		lm.locationX = (float*)malloc(sizeof(float)*len);
		CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_lm->locationX, sizeof(float*), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(lm.locationX, d_bufferPtr, sizeof(float)*len, cudaMemcpyDeviceToHost));
		lm.locationY = (float*)malloc(sizeof(float)* len);
		CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_lm->locationY, sizeof(float*), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(lm.locationY, d_bufferPtr, sizeof(float)*len, cudaMemcpyDeviceToHost));
#ifdef _3D
		lm.locationZ = (float*)malloc(sizeof(float)*len);
		CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_lm->locationZ, sizeof(float*), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(lm.locationZ, d_bufferPtr, sizeof(float)*len, cudaMemcpyDeviceToHost));
#endif
#endif
		oFile << len << "\n";
		for (int i = 0; i < len; i++)
		{
#ifdef AOS_MESSAGES
#if defined(_2D)
            sprintf(&buffer[0], "%.9g,%.9g\n", lm.location[i].x, lm.location[i].y);
#elif defined(_3D)
            sprintf(&buffer[0], "%.9g,%.9g,%.9g\n", lm.location[i].x, lm.location[i].y, lm.location[i].z);
#endif
#else
#if defined(_2D)
            sprintf(&buffer[0], "%.9g,%.9g\n", lm.locationX[i], lm.locationY[i]);
#elif defined(_3D)
			sprintf(&buffer[0], "%.9g,%.9g,%.9g\n", lm.locationX[i], lm.locationY[i], lm.locationZ[i]);
#endif
#endif
			oFile << buffer;
		}
#ifdef AOS_MESSAGES
        free(lm.location)
#else
        free(lm.locationX);
        free(lm.locationY);
#ifdef _3D
        free(lm.locationZ);
#endif
#endif
		oFile.close();
	}
	else
	{
		printf("Failed to open file: %s\n", path);
	}
}

void exportNullAgents(std::shared_ptr<SpatialPartition> s, const char *path, const DIMENSIONS_VEC *results)
{

    int len = s->getLocationCount();
    std::ofstream oFile;
    oFile.open(path);
    char buffer[1024];
    if (oFile.is_open())
    {
        //allocate
        float *d_bufferPtr;
        LocationMessages *d_lm = s->d_getLocationMessages();
        LocationMessages lm;
#ifdef AOS_MESSAGES
        lm.location = (DIMENSIONS_VEC*)malloc(sizeof(DIMENSIONS_VEC)*len);
        CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_lm->location, sizeof(DIMENSIONS_VEC*), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(lm.location, d_bufferPtr, sizeof(DIMENSIONS_VEC)*len, cudaMemcpyDeviceToHost));
#else
        lm.locationX = (float*)malloc(sizeof(float)*len);
        CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_lm->locationX, sizeof(float*), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(lm.locationX, d_bufferPtr, sizeof(float)*len, cudaMemcpyDeviceToHost));
        lm.locationY = (float*)malloc(sizeof(float)* len);
        CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_lm->locationY, sizeof(float*), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(lm.locationY, d_bufferPtr, sizeof(float)*len, cudaMemcpyDeviceToHost));
#ifdef _3D
        lm.locationZ = (float*)malloc(sizeof(float)*len);
        CUDA_CALL(cudaMemcpy(&d_bufferPtr, &d_lm->locationZ, sizeof(float*), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(lm.locationZ, d_bufferPtr, sizeof(float)*len, cudaMemcpyDeviceToHost));
#endif
#endif
        oFile << len << "\n";
        for (int i = 0; i < len; i++)
        {
#ifdef AOS_MESSAGES
#if defined(_2D)
            sprintf(&buffer[0], "%.9g,%.9g,%.9g,%.9g\n", lm.location[i].x, lm.location[i].y, results[i].x, results[i].y);
#elif defined(_3D)
            sprintf(&buffer[0], "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", lm.location[i].x, lm.location[i].y, lm.location[i].z, results[i].x, results[i].y, results[i].z);
#endif
#else
#if defined(_2D)
            sprintf(&buffer[0], "%.9g,%.9g,%.9g,%.9g\n", lm.locationX[i], lm.locationY[i], results[i].x, results[i].y);
#elif defined(_3D)
            sprintf(&buffer[0], "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", lm.locationX[i], lm.locationY[i], lm.locationZ[i], results[i].x, results[i].y, results[i].z);
#endif
#endif
            oFile << buffer;
        }
#ifdef AOS_MESSAGES
        free(lm.location)
#else
        free(lm.locationX);
        free(lm.locationY);
#ifdef _3D
        free(lm.locationZ);
#endif
#endif
        oFile.close();
    }
    else
    {
        printf("Failed to open file: %s\n", path);
    }
}

/**
 * Export steps timing in csv format
 */
void exportSteps(const int argc, char **argv, const Time_Step *ts, const NeighbourhoodStats *ns, const unsigned int &stepCount, const char *path)
{
    std::ofstream oFile;
    oFile.open(path);
    char buffer[1024];
    if (oFile.is_open())
    {
        //Header row 1: runtime args
        for (unsigned int i = 0; i < argc;++i)
        {
            oFile << argv[i];
            oFile << " ";
        }
        oFile << "\n";
        //Header row 2: high lvl titles
        oFile << ",";
        oFile << "step avg (s),,,";
        oFile << "Moore Neighbourhood Sizes (s),,,,";
        oFile << "\n";
        //Header row 3: actual titles
        oFile << "Iteration,";
        oFile << "overall,";
        oFile << "kernel,";
        oFile << "texture,";
        oFile << "min,";
        oFile << "max,";
        oFile << "avg,";
        oFile << "std deviation,";
        oFile << "\n";
        //Data
        for (unsigned int i = 0; i < stepCount;++i)
        {
            sprintf(&buffer[0], "%u,", i);
            oFile << buffer;
            sprintf(&buffer[0], "%.9g,%.9g,%.9g,", ts[i].overall, ts[i].kernel, ts[i].texture);
            oFile << buffer;
            sprintf(&buffer[0], "%u,%u,%.9g,%.9g,", ns[i].min, ns[i].max, ns[i].average, ns[i].standardDeviation);
            oFile << buffer;
            oFile << "\n";
        }
        oFile.close();
    }
    else
    {
        printf("Failed to open file: %s\n", path);
    }
}