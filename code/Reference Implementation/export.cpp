#include "export.h"
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>

#include "rapidxml/rapidxml.hpp"
#include "rapidxml/rapidxml_print.hpp"

void exportPopulation(SpatialPartition* s, ModelParams *model, char *path)
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
	//states
	rapidxml::xml_node<> *states_node = doc.allocate_node(rapidxml::node_element, states_node_str);
	doc.append_node(states_node);
	//itno
	rapidxml::xml_node<> *itno_node = doc.allocate_node(rapidxml::node_element, itno_node_str, doc.allocate_string("0"));
	states_node->append_node(itno_node);
	//model params
	rapidxml::xml_node<> *params_node = doc.allocate_node(rapidxml::node_element, params_node_str);
	{
		//width
		sprintf(buffer, "%d", model->width);
		rapidxml::xml_node<> *width_node = doc.allocate_node(rapidxml::node_element, width_node_str, doc.allocate_string(buffer));
		params_node->append_node(width_node);
		//seed
		sprintf(buffer, "%f", model->interactionRad);
		rapidxml::xml_node<> *rad_node = doc.allocate_node(rapidxml::node_element, rad_node_str, doc.allocate_string(buffer));
		params_node->append_node(rad_node);
		//density
		sprintf(buffer, "%f", model->density);
		rapidxml::xml_node<> *density_node = doc.allocate_node(rapidxml::node_element, density_node_str, doc.allocate_string(buffer));
		params_node->append_node(density_node);
		//seed
		sprintf(buffer, "%llu", model->seed);
		rapidxml::xml_node<> *seed_node = doc.allocate_node(rapidxml::node_element, seed_node_str, doc.allocate_string(buffer));
		params_node->append_node(seed_node);
		//att force
		sprintf(buffer, "%f", model->attractionForce);
		rapidxml::xml_node<> *attract_node = doc.allocate_node(rapidxml::node_element, attract_node_str, doc.allocate_string(buffer));
		params_node->append_node(attract_node);
		//rep force
		sprintf(buffer, "%f", model->repulsionForce);
		rapidxml::xml_node<> *repel_node = doc.allocate_node(rapidxml::node_element, repel_node_str, doc.allocate_string(buffer));
		params_node->append_node(repel_node);
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

			sprintf(buffer, "%f", lm.locationX[i]);
			rapidxml::xml_node<> *x_node = doc.allocate_node(rapidxml::node_element, x_node_str, doc.allocate_string(buffer));
			xagent_node->append_node(x_node);

			sprintf(buffer, "%f", lm.locationY[i]);
			rapidxml::xml_node<> *y_node = doc.allocate_node(rapidxml::node_element, y_node_str, doc.allocate_string(buffer));
			xagent_node->append_node(y_node);

			sprintf(buffer, "%f", lm.locationZ[i]);
			rapidxml::xml_node<> *z_node = doc.allocate_node(rapidxml::node_element, z_node_str, doc.allocate_string(buffer));
			xagent_node->append_node(z_node);

			rapidxml::xml_node<> *fx_node = doc.allocate_node(rapidxml::node_element, fx_node_str, zero_pt_zero_str);
			xagent_node->append_node(fx_node);

			rapidxml::xml_node<> *fy_node = doc.allocate_node(rapidxml::node_element, fy_node_str, zero_pt_zero_str);
			xagent_node->append_node(fy_node);

			rapidxml::xml_node<> *fz_node = doc.allocate_node(rapidxml::node_element, fz_node_str, zero_pt_zero_str);
			xagent_node->append_node(fz_node);
		}
		states_node->append_node(xagent_node);
	}

	//Actually do the output
	std::ofstream f;
	f.open(outFile);

	f << doc;

	f.close();
	free(lm.locationX);
	free(lm.locationY);
	free(lm.locationZ);
}