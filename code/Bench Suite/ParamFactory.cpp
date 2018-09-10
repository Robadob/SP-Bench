#define _CRT_SECURE_NO_WARNINGS
#include "ParamFactory.h"

#include <cstdio>

#include "rapidjson/filereadstream.h"
#include "rapidjson/document.h"

template <typename Encoding, typename Allocator = rapidjson::MemoryPoolAllocator<>, typename StackAllocator = rapidjson::CrtAllocator>
unsigned int parseSteps(rapidjson::GenericValue<Encoding, Allocator> &valueType);
template <typename Encoding, typename Allocator = rapidjson::MemoryPoolAllocator<>, typename StackAllocator = rapidjson::CrtAllocator>
ParamSet parseConfigSet(rapidjson::GenericValue<Encoding, Allocator> &valueType);
template <typename Encoding, typename Allocator = rapidjson::MemoryPoolAllocator<>, typename StackAllocator = rapidjson::CrtAllocator>
std::shared_ptr<ModelParams> parseConfig(rapidjson::GenericValue<Encoding, Allocator> &valueType, std::shared_ptr<ModelParams> _default = nullptr);

std::vector<ParamSet> ParamFactory::read(const char* inputFile)
{
	std::vector<ParamSet> rtn;
	FILE* fp = fopen(inputFile, "rb"); // non-Windows use "r"
	if (!fp)
	{
		printf("Failed to open bench config: %s\n", inputFile);
		return rtn;
	}
	//Open stream to file
	char readBuffer[65536];
	rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document d;
	d.ParseStream(is);
	fclose(fp);
	//Parse ModelParms
	if (d.IsObject())
	{//Single configset
		ParamSet ptr = parseConfigSet(d);
		if (ptr.validate())
		{
			printf("Parsed 1/1 config set in file '%s'\n", inputFile);
			rtn.push_back(ptr);
		}
		else
			printf("Unable to parse config set in file '%s'.\n", inputFile);
	}
	else if (d.IsArray())
	{//Multi configset
		for (unsigned int i = 0; i < d.Size();++i)
		{
			ParamSet ptr = parseConfigSet(d[i]);
			if (ptr.validate())
			{
				rtn.push_back(ptr);
			}			
		}
		if (rtn.size())
		{
			printf("Parsed %lu/%u config sets in file '%s'\n", rtn.size(), d.Size(), inputFile);
		}
		else
			printf("Unable to parse config sets in file '%s'.\n", inputFile);
		
	}
	//Return
	return rtn;
}

void ParamFactory::write(std::vector<std::shared_ptr<ModelParams>> params, const char* outputFile)
{
}

template <typename Encoding, typename Allocator, typename StackAllocator>
ParamSet parseConfigSet(rapidjson::GenericValue<Encoding, Allocator> &valueType)
{
	ParamSet rtn = ParamSet();
	//Read name
	if (valueType.HasMember("name") && valueType["name"].IsString())
		rtn.name = valueType["name"].GetString();
	//Read start
	if (valueType.HasMember("start"))
		rtn.start = parseConfig(valueType["start"]);
	else if (valueType.HasMember("enumerator"))
		rtn.start = parseConfig(valueType);
	if (!rtn.start)
		return ParamSet();
	//Read end/end1
	if (valueType.HasMember("end"))
	{
		rtn.end1 = parseConfig(valueType["end"], rtn.start);
		rtn.steps1 = parseSteps(valueType["end"]);
	}
	else if (valueType.HasMember("end1"))
	{
		rtn.end1 = parseConfig(valueType["end1"], rtn.start);
		rtn.steps1 = parseSteps(valueType["end1"]);
	}
	//Read end2
	if (rtn.end1&&valueType.HasMember("end2"))
	{
		rtn.end2 = parseConfig(valueType["end2"], rtn.end1);
		rtn.steps2 = parseSteps(valueType["end2"]);
	}
	return rtn;
}
template <typename Encoding, typename Allocator, typename StackAllocator>
std::shared_ptr<ModelParams> parseConfig(rapidjson::GenericValue<Encoding, Allocator> &valueType, std::shared_ptr<ModelParams> _default)
{
	//Detect config type
	if (!(valueType.HasMember("enumerator") && valueType["enumerator"].IsInt())&&!_default)
		return nullptr;
	//Allocate and parse specific type
	std::shared_ptr<ModelParams> rtn;
    switch (_default?_default->enumerator():ModelEnum(valueType["enumerator"].GetInt()))
	{
	case Null:
	{
		auto a = std::make_shared<NullParams>();
		if (auto b = std::dynamic_pointer_cast<NullParams>(_default))
			a->operator=(*b);
		{//Parse null config
			//Agents
			if (valueType.HasMember("agents") && valueType["agents"].IsInt())
				a->agents = (unsigned int)valueType["agents"].GetInt();
			else if (!_default)
				fprintf(stderr, "Warning: Property 'agents' missing from null config, default value '%u' used.\n", a->agents);
			//Density
			if (valueType.HasMember("density") && valueType["density"].IsFloat())
                a->density = valueType["density"].GetFloat();
            else if (valueType.HasMember("density") && valueType["density"].IsInt())
                a->density = (float)valueType["density"].GetInt();
			else if(!_default)
				fprintf(stderr, "Warning: Property 'density' missing from null config, default value '%g' used.\n", a->density);
		}
		rtn = a;
		break;
	}
	case Circles:
	{
		auto a = std::make_shared<CirclesParams>();
		if (auto b = std::dynamic_pointer_cast<CirclesParams>(_default))
			a->operator=(*b);
		{//Parse circles config
			//Agents
			if (valueType.HasMember("agents") && valueType["agents"].IsInt())
				a->agents = (unsigned int)valueType["agents"].GetInt();
			else if(!_default)
				fprintf(stderr, "Warning: Property 'agents' missing from circles config, default value '%u' used.\n", a->agents);
			//Density
            if (valueType.HasMember("density") && valueType["density"].IsFloat())
                a->density = valueType["density"].GetFloat();
            else if (valueType.HasMember("density") && valueType["density"].IsInt())
                a->density = (float)valueType["density"].GetInt();
			else if (!_default)
				fprintf(stderr, "Warning: Property 'density' missing from circles config, default value '%g' used.\n", a->density);
			//ForceModifier
			if (valueType.HasMember("forceModifier") && valueType["forceModifier"].IsFloat())
                a->forceModifier = valueType["forceModifier"].GetFloat();
            else if (valueType.HasMember("forceModifier") && valueType["forceModifier"].IsInt())
                a->forceModifier = (float)valueType["forceModifier"].GetInt();
			else if (!_default)
				fprintf(stderr, "Warning: Property 'forceModifier' missing from circles config, default value '%g' used.\n", a->forceModifier);
		}
		rtn = a;
		break;
	}
	case Density:
	{
		auto a = std::make_shared<DensityParams>();
		if (auto b = std::dynamic_pointer_cast<DensityParams>(_default))
			a->operator=(*b);
		{//Parse density config
			//AgentsPerCluster
			if (valueType.HasMember("agentsPerCluster") && valueType["agentsPerCluster"].IsInt())
				a->agentsPerCluster = (unsigned int)valueType["agentsPerCluster"].GetInt();
			else if (!_default)
				fprintf(stderr, "Warning: Property 'agentsPerCluster' missing from density config, default value '%u' used.\n", a->agentsPerCluster);
			//EnvWidth
			if (valueType.HasMember("envWidth") && valueType["envWidth"].IsFloat())
				a->envWidth = valueType["envWidth"].GetFloat();
			else if (!_default)
				fprintf(stderr, "Warning: Property 'envWidth' missing from density config, default value '%g' used.\n", a->interactionRad);
			//InteractionRad
			if (valueType.HasMember("interactionRad") && valueType["interactionRad"].IsFloat())
				a->interactionRad = valueType["interactionRad"].GetFloat();
			else if (!_default)
				fprintf(stderr, "Warning: Property 'interactionRad' missing from density config, default value '%g' used.\n", a->interactionRad);
			//ClusterCount
			if (valueType.HasMember("clusterCount") && valueType["clusterCount"].IsInt())
				a->clusterCount = (unsigned int)valueType["clusterCount"].GetInt();
			else if (!_default)
				fprintf(stderr, "Warning: Property 'clusterCount' missing from density config, default value '%u' used.\n", a->clusterCount);
			//ClusterRad
			if (valueType.HasMember("clusterRad") && valueType["clusterRad"].IsFloat())
				a->clusterRad = valueType["clusterRad"].GetFloat();
			else if (!_default)
				fprintf(stderr, "Warning: Property 'clusterRad' missing from density config, default value '%g' used.\n", a->clusterRad);
			//UniformDensity
			if (valueType.HasMember("uniformDensity") && valueType["uniformDensity"].IsFloat())
				a->uniformDensity = valueType["uniformDensity"].GetFloat();
			else if (!_default)
				fprintf(stderr, "Warning: Property 'uniformDensity' missing from density config, default value '%g' used.\n", a->uniformDensity);
		}
		rtn = a;
		break;
	}
	default: 
		return nullptr;
	}
	//Parse generics
	{
		//Iterations
		if (valueType.HasMember("iterations") && valueType["iterations"].IsInt())
			rtn->iterations = valueType["iterations"].GetInt();
		else if (!_default)
			fprintf(stderr, "Warning: Property 'iterations' missing from config, default value '%llu' used.\n", rtn->iterations);
		//Seed
		if (valueType.HasMember("seed") && valueType["seed"].IsInt())
			rtn->seed = valueType["seed"].GetInt();
		else if (!_default)
			fprintf(stderr, "Warning: Property 'seed' missing from config, default value '%llu' used.\n", rtn->seed);
	}
	return rtn;
}
template <typename Encoding, typename Allocator, typename StackAllocator>
unsigned int parseSteps(rapidjson::GenericValue<Encoding, Allocator> &valueType)
{
	if (valueType.HasMember("steps") && valueType["steps"].IsInt())
		return valueType["steps"].GetInt();
	return 0;
}