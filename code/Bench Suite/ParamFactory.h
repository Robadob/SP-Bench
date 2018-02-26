#ifndef __ParamFactory_h__
#define __ParamFactory_h__
#include <memory>
#include <vector>
#include "ParamSet.h"

class ParamFactory
{
public:
	static std::vector<ParamSet>read(const char *inputFile);

	static void write(std::vector<std::shared_ptr<ModelParams>> params, const char *outputFile);
private:


};
#endif //__ParamFactory_h__