#include "ParamFactory.h"

#ifdef _MSC_VER
const char *BIN_X64 = "../bin/x64/";
const char *OUT_DIR = "./out/";
#else
const char *BIN_X64 = "./";
const char *OUT_DIR = "../../out/";
#endif

int main(int argc, char* argv[])
{
	//Set configs
	ParamSet::setBinDir(BIN_X64);
	ParamSet::setOutputDir(OUT_DIR);

	//Iterate files
	for (int i = 1; i < argc; i++)
	{
		//Load file
		std::vector<ParamSet> a = ParamFactory::read(argv[i]);
		//Iterate loaded params
		for (const ParamSet &b : a)
		{
			printf("Running %s\n", b.name.c_str());
			b.execute();
		}
	}
}