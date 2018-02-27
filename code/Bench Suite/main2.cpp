#include "ParamFactory.h"

#include <algorithm>
#include <string>

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
    ParamSet::setDeviceId(0);

    //Iterate args
    int i = 1;
	for (; i < argc; i++)
	{
        //Get arg as lowercase
		std::string arg(argv[i]);
		std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);
        
        //-device <uint>, Uses the specified cuda device, defaults to 0
		if (arg.compare("-device") == 0)
		{
            unsigned int _device = (unsigned int)strtoul(argv[++i], nullptr, 0);
            ParamSet::setDeviceId(_device);
            printf("Using device '%u'.\n", _device);
            continue;
		}
        break;//Only iterate loop manually (if we handled an arg)
	}
    
	//Iterate files
	for (; i < argc; i++)
    {
		//Load file
		std::vector<ParamSet> a = ParamFactory::read(argv[i]);
		//Iterate loaded params
		for (const ParamSet &b : a)
		{
			b.execute();
		}
    }
}