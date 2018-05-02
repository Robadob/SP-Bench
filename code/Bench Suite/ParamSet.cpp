#include "ParamSet.h"
#ifdef _MSC_VER
#include <windows.h>
#define popen _popen
#define pclose _pclose
#else
//#include <filesystem>
#include <sys/stat.h>
#endif

std::string ParamSet::BIN_DIR = "./";
std::string ParamSet::OUT_DIR = "./OUT/";
unsigned int ParamSet::DEVICE_ID = 0;
#ifdef _MSC_VER
std::vector<Executable> ParamSet::BINARIES = { { "Default", "Release-Mod-Default.exe" }, { "Strips", "Release-Mod-Strips.exe" }, { "Modular", "Release-Mod-Modular.exe" }, { "ModularStrips", "Release-Mod-ModularStrips.exe" } };
#else
std::vector<Executable> ParamSet::BINARIES = { { "Default", "release-mod-default" }, { "Strips", "release-mod-strips" }, { "Modular", "release-mod-modular" }, { "ModularStrips", "release-mod-modularstrips3d" } };
#endif

void ParamSet::setBinDir(const char*d)
{
    BIN_DIR = d;
}
void ParamSet::setOutputDir(const char*d)
{
    OUT_DIR = d;
}
void ParamSet::setDeviceId(unsigned int d)
{
    DEVICE_ID = d;
}

bool ParamSet::validate() const
{
    if (!start)
        return false;
    if (end1&&start->enumerator() != end1->enumerator())
        return false;
    if (end1&&end2&&end1->enumerator() != end2->enumerator())
        return false;
    if ((!end1) && end2)
        return false;
    if (end1 && !steps1)
        return false;
    if (end2 && !steps2)
        return false;
    return true;
}
void ParamSet::execute() const
{
    //Validate
    if (!validate())
    {
        fprintf(stderr, "Unable to execute invalid param set '%s'.\n", name.c_str());
        return;
    }
    //Execute
    {
        //Detect type of run required
        if (end2&&end1&&start)
        {//2D Sweep
            runCollated2D();
        }
        else if (end1&&start)
        {//1D Sweep
            runCollated();
        }
        else if (start)
        {//Single run
            run();
        }
    }
}
#include "main.cpp"
void ParamSet::run() const
{
    printf("Running %s\n", name.c_str());
    //Create log
    FILE *log_F = createLogFile(std::to_string(time(nullptr)).c_str());
    if (log_F)
    {
        //Write header
        switch (start->enumerator())
        {
        case Null:
            logCollatedHeader(log_F, *std::dynamic_pointer_cast<NullParams>(start));
            break;
        case Circles:
            logCollatedHeader(log_F, *std::dynamic_pointer_cast<CirclesParams>(start));
            break;
        case Density:
            logCollatedHeader(log_F, *std::dynamic_pointer_cast<DensityParams>(start));
            break;
        default:
            fprintf(stderr, "Error!\n");
            return;
        }
        //Output structs
        unsigned int agentCount;
        Time_Init initRes;
        Time_Step_dbl stepRes;
        float totalTime;
        NeighbourhoodStats nsFirst, nsLast;
        //Loop models
        for (const auto &bin : BINARIES)
        {
            //Clear output structures
            memset(&stepRes, 0, sizeof(Time_Step_dbl));
            memset(&initRes, 0, sizeof(Time_Init));
            //Bench
            bool success = false;
            switch (start->enumerator())
            {
            case Null:
            {
                NullParams n;
                success = executeBenchmark(bin.file, *std::dynamic_pointer_cast<NullParams>(start), &n, &agentCount, &initRes, &stepRes, &totalTime, &nsFirst, &nsLast);
            }
                break;
            case Circles:
            {
                CirclesParams n;
                success = executeBenchmark(bin.file, *std::dynamic_pointer_cast<CirclesParams>(start), &n, &agentCount, &initRes, &stepRes, &totalTime, &nsFirst, &nsLast);
            }
                break;
            case Density:
            {
                DensityParams n;
                success = executeBenchmark(bin.file, *std::dynamic_pointer_cast<DensityParams>(start), &n, &agentCount, &initRes, &stepRes, &totalTime, &nsFirst, &nsLast);
            }
                break;
            default:
                fprintf(stderr, "Error!\n");
                return;
            }
            if (!success)
            {
                fprintf(stderr, "\rBenchmark '%s' '%s', exiting early.\n", name.c_str(), bin.name);
                return;
            }
            //Write result body
            log(log_F, &initRes, &stepRes, totalTime);
        }
        //Write run config
        //AgentCount
        log(log_F, agentCount);
        //Neighbourhood stats
        log(log_F, &nsFirst);
        log(log_F, &nsLast);
        //Model Args
        switch (start->enumerator())
        {
        case Null:
            log(log_F, std::dynamic_pointer_cast<NullParams>(start).get());
            break;
        case Circles:
            log(log_F, std::dynamic_pointer_cast<CirclesParams>(start).get());
            break;
        case Density:
            log(log_F, std::dynamic_pointer_cast<DensityParams>(start).get());
            break;
        default:
            fprintf(stderr, "Error!\n");
            return;
        }
        //Flush
        fputs("\n", log_F);
        fflush(log_F);
        //Close log
        fclose(log_F);
        //Print confirmation to console
        printf("\rCompleted run %s              \n", name.c_str());
    }
}
void ParamSet::runCollated() const
{
    printf("Running(1D) %s\n", name.c_str());
    //Create log
    FILE *log_F = createLogFile(std::to_string(time(nullptr)).c_str());
    if (log_F)
    {
        //Write header
        switch (start->enumerator())
        {
        case Null:
            logCollatedHeader(log_F, *std::dynamic_pointer_cast<NullParams>(start));
            break;
        case Circles:
            logCollatedHeader(log_F, *std::dynamic_pointer_cast<CirclesParams>(start));
            break;
        case Density:
            logCollatedHeader(log_F, *std::dynamic_pointer_cast<DensityParams>(start));
            break;
        default:
            fprintf(stderr, "Error!\n");
            return;
        }
        unsigned int currentRuns = 0;
        //Output structs
        unsigned int agentCount;
        Time_Init initRes;
        Time_Step_dbl stepRes;
        float totalTime;
        NeighbourhoodStats nsFirst, nsLast;
        //Loop models
        for (unsigned int i = 0; i < steps1; i++)
        {
            std::shared_ptr<ModelParams> lerpArgs = interpolateParams(start, end1, i, steps1);
            for (const auto &bin : BINARIES)
            {
                //Clear output structures
                memset(&stepRes, 0, sizeof(Time_Step_dbl));
                memset(&initRes, 0, sizeof(Time_Init));
                //Bench
                bool success = false;
                switch (start->enumerator())
                {
                case Null:
                {
                    NullParams n;
                    success = executeBenchmark(bin.file, *std::dynamic_pointer_cast<NullParams>(lerpArgs), &n, &agentCount, &initRes, &stepRes, &totalTime, &nsFirst, &nsLast);
                }
                    break;
                case Circles:
                {
                    CirclesParams n;
                    success = executeBenchmark(bin.file, *std::dynamic_pointer_cast<CirclesParams>(lerpArgs), &n, &agentCount, &initRes, &stepRes, &totalTime, &nsFirst, &nsLast);
                }
                    break;
                case Density:
                {
                    DensityParams n;
                    success = executeBenchmark(bin.file, *std::dynamic_pointer_cast<DensityParams>(lerpArgs), &n, &agentCount, &initRes, &stepRes, &totalTime, &nsFirst, &nsLast);
                }
                    break;
                default:
                    fprintf(stderr, "Error!\n");
                    return;
                }
                if (!success)
                {
                    fprintf(stderr, "\rBenchmark '%s' '%s', exiting early.\n", name.c_str(), bin.name);
                    return;
                }
                //Write result body
                log(log_F, &initRes, &stepRes, totalTime);
            }
            printf("\rRun %u/%u", currentRuns++, steps1);
            //Write run config
            //AgentCount
            log(log_F, agentCount);
            //Neighbourhood stats
            log(log_F, &nsFirst);
            log(log_F, &nsLast);
            //Model Args
            switch (start->enumerator())
            {
            case Null:
                log(log_F, std::dynamic_pointer_cast<NullParams>(lerpArgs).get());
                break;
            case Circles:
                log(log_F, std::dynamic_pointer_cast<CirclesParams>(lerpArgs).get());
                break;
            case Density:
                log(log_F, std::dynamic_pointer_cast<DensityParams>(lerpArgs).get());
                break;
            default:
                fprintf(stderr, "Error!\n");
                return;
            }
            //Flush
            fputs("\n", log_F);
            fflush(log_F);
        }
        //Close log
        fclose(log_F);
        //Print confirmation to console
        printf("\rCompleted run %s              \n", name.c_str());
    }
}
void ParamSet::runCollated2D() const
{
    printf("Running(2D) %s\n", name.c_str());
    //Create log
    FILE *log_F = createLogFile(std::to_string(time(nullptr)).c_str());
    if (log_F)
    {
        //Write header
        switch (start->enumerator())
        {
        case Null:
            logCollatedHeader(log_F, *std::dynamic_pointer_cast<NullParams>(start));
            break;
        case Circles:
            logCollatedHeader(log_F, *std::dynamic_pointer_cast<CirclesParams>(start));
            break;
        case Density:
            logCollatedHeader(log_F, *std::dynamic_pointer_cast<DensityParams>(start));
            break;
        default:
            fprintf(stderr, "Error!\n");
            return;
        }
        const unsigned int totalRuns = steps1 * steps2;
        unsigned int currentRuns = 0;
        //Output structs
        unsigned int agentCount;
        Time_Init initRes;
        Time_Step_dbl stepRes;
        float totalTime;
        NeighbourhoodStats nsFirst, nsLast;
        //Loop models
        for (unsigned int i = 0; i < steps1; i++)
        {
            for (unsigned int j = 0; j < steps2; j++)
            {
                std::shared_ptr<ModelParams> lerpArgs = interpolateParams2D(start, end1, end2, i, steps1, j, steps2);
                for (const auto &bin : BINARIES)
                {
                    //Clear output structures
                    memset(&stepRes, 0, sizeof(Time_Step_dbl));
                    memset(&initRes, 0, sizeof(Time_Init));
                    //Bench
                    bool success = false;
                    switch (start->enumerator())
                    {
                    case Null:
                    {
                        NullParams n;
                        success = executeBenchmark(bin.file, *std::dynamic_pointer_cast<NullParams>(lerpArgs), &n, &agentCount, &initRes, &stepRes, &totalTime, &nsFirst, &nsLast);
                    }
                        break;
                    case Circles:
                    {
                        CirclesParams n;
                        success = executeBenchmark(bin.file, *std::dynamic_pointer_cast<CirclesParams>(lerpArgs), &n, &agentCount, &initRes, &stepRes, &totalTime, &nsFirst, &nsLast);
                    }
                        break;
                    case Density:
                    {
                        DensityParams n;
                        success = executeBenchmark(bin.file, *std::dynamic_pointer_cast<DensityParams>(lerpArgs), &n, &agentCount, &initRes, &stepRes, &totalTime, &nsFirst, &nsLast);
                    }
                        break;
                    default:
                        fprintf(stderr, "Error!\n");
                        return;
                    }
                    if (!success)
                    {
                        fprintf(stderr, "\rBenchmark '%s' '%s', exiting early.\n", name.c_str(), bin.name);
                        return;
                    }
                    //Write result body
                    log(log_F, &initRes, &stepRes, totalTime);
                }
                printf("\rRun %u/%u", currentRuns++, totalRuns);
                //Write run config
                //AgentCount
                log(log_F, agentCount);
                //Neighbourhood stats
                log(log_F, &nsFirst);
                log(log_F, &nsLast);
                //Model Args
                switch (start->enumerator())
                {
                case Null:
                    log(log_F, std::dynamic_pointer_cast<NullParams>(lerpArgs).get());
                    break;
                case Circles:
                    log(log_F, std::dynamic_pointer_cast<CirclesParams>(lerpArgs).get());
                    break;
                case Density:
                    log(log_F, std::dynamic_pointer_cast<DensityParams>(lerpArgs).get());
                    break;
                default:
                    fprintf(stderr, "Error!\n");
                    return;
                }
                //Flush
                fputs("\n", log_F);
                fflush(log_F);
            }
        }
        //Close log
        fclose(log_F);
        //Print confirmation to console
        printf("\rCompleted run %s              \n", name.c_str());
    }
}
FILE *ParamSet::createLogFile(const char*timestamp) const
{
    FILE *log_F = nullptr;
    std::string logPath(OUT_DIR);
#ifdef _MSC_VER
    CreateDirectory(logPath.c_str(), NULL);
#else
    mkdir(logPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
    logPath = logPath.append(name.c_str());
    logPath = logPath.append("/");
#ifdef _MSC_VER
    CreateDirectory(logPath.c_str(), NULL);
#else
    mkdir(logPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    //if (!exists(logPath)) { // Check if folder exists
    //    create_directory(logPath.c_str()); // create folder
    //}
#endif
    logPath = logPath.append("collated");
    logPath = logPath.append(std::to_string(time(0)));
    logPath = logPath.append(".csv");
    log_F = fopen(logPath.c_str(), "w");
    if (!log_F)
    {
        fprintf(stderr, "Benchmark '%s' failed to create log file '%s'\n", name.c_str(), logPath.c_str());
    }
    return log_F;
}

//Collated versions
void ParamSet::logCollatedHeader(FILE *out, const CirclesParams &modelArgs)
{
    //Row 1
    for (const auto &bin : BINARIES)
    {//9 columns per model
        fprintf(out, "%s,,,,,,,,,", bin.name);
    }
    fputs(",", out);//Agent count
    fputs(",,,,,,,,", out);//Neighbourhood stats
    //fputs(",,,,,,", out);//Model Args
    fputs(",,,,", out);//Model Args
    fputs("\n", out);
    //Row 2
    for (const auto &bin : BINARIES)
    {//9 columns per model
        fputs("init (s)", out);
        fputs(",,,,,", out);
        fputs("step avg (s)", out);
        fputs(",,,", out);
        fputs("overall (s)", out);
        fputs(",", out);
    }
    fputs(",", out);//Agent count
    fputs("Neighbourhood Stats", out);
    fputs(",,,,,,,,", out);
    fputs("Model", out);
    //fputs(",,,,,,,", out);
    fputs(",,,,,", out);
    fputs("\n", out);
    //Row 3
    for (const auto &bin : BINARIES)
    {//9 columns per model
        //Init
        fputs("overall", out);
        fputs(",", out);
        fputs("initCurand", out);
        fputs(",", out);
        fputs("kernel", out);
        fputs(",", out);
        fputs("pbm", out);
        fputs(",", out);
        fputs("freeCurand", out);
        fputs(",", out);
        //Step avg
        fputs("overall", out);
        fputs(",", out);
        fputs("kernel", out);
        fputs(",", out);
        fputs("texture", out);
        fputs(",", out);
        //Total
        fputs("time", out);
        fputs(",", out);
    }
    fputs("Agent Count,", out);
    //Neighbourhood stats
    fputs("First Min", out);
    fputs(",", out);
    fputs("First Max", out);
    fputs(",", out);
    fputs("First Avg", out);
    fputs(",", out);
    fputs("First SD", out);
    fputs(",", out);
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
    fputs(",", out);
    fputs("Last SD", out);
    fputs(",", out);
    //ModelArgs
    //fputs("width", out);
    fputs("agents", out);
    fputs(",", out);
    fputs("density", out);
    fputs(",", out);
    //fputs("interactionRad", out);
    //fputs(",", out);
    //fputs("attractionForce", out);
    //fputs(",", out);
    //fputs("repulsionForce", out);
    fputs("forceModifier", out);
    fputs(",", out);
    fputs("iterations", out);
    fputs(",", out);
    fputs("seed", out);
    fputs(",", out);
    fputs("\n", out);
    fflush(out);
}
void ParamSet::logCollatedHeader(FILE *out, const NullParams &modelArgs)
{
    //Row 1
    for (const auto &bin : BINARIES)
    {//9 columns per model
        fprintf(out, "%s,,,,,,,,,", bin.name);
    }
    fputs(",", out);//Agent count
    fputs(",,,,,,,,", out);//Neighbourhood stats
    fputs(",,,,,", out);//Model Args
    fputs("\n", out);
    //Row 2
    for (const auto &bin : BINARIES)
    {//9 columns per model
        fputs("init (s)", out);
        fputs(",,,,,", out);
        fputs("step avg (s)", out);
        fputs(",,,", out);
        fputs("overall (s)", out);
        fputs(",", out);
    }
    fputs(",", out);//Agent count
    fputs("Neighbourhood Stats", out);
    fputs(",,,,,,,,", out);
    fputs("Model", out);
    fputs(",,,,,", out);
    fputs("\n", out);
    //Row 3
    for (const auto &bin : BINARIES)
    {//9 columns per model
        //Init
        fputs("overall", out);
        fputs(",", out);
        fputs("initCurand", out);
        fputs(",", out);
        fputs("kernel", out);
        fputs(",", out);
        fputs("pbm", out);
        fputs(",", out);
        fputs("freeCurand", out);
        fputs(",", out);
        //Step avg
        fputs("overall", out);
        fputs(",", out);
        fputs("kernel", out);
        fputs(",", out);
        fputs("texture", out);
        fputs(",", out);
        //Total
        fputs("time", out);
        fputs(",", out);
    }
    fputs("Agent Count,", out);
    //Neighbourhood stats
    fputs("First Min", out);
    fputs(",", out);
    fputs("First Max", out);
    fputs(",", out);
    fputs("First Avg", out);
    fputs(",", out);
    fputs("First SD", out);
    fputs(",", out);
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
    fputs(",", out);
    fputs("Last SD", out);
    fputs(",", out);
    fflush(out);
    //ModelArgs
    fputs("agents-in", out);
    fputs(",", out);
    fputs("density", out);
    fputs(",", out);
    fputs("iterations", out);
    fputs(",", out);
    fputs("seed", out);
    fputs(",", out);
    fputs("\n", out);
    fflush(out);
}
void ParamSet::logCollatedHeader(FILE *out, const DensityParams &modelArgs)
{
    //Row 1
    for (const auto &bin : BINARIES)
    {//9 columns per model
        fprintf(out, "%s,,,,,,,,,", bin.name);
    }
    fputs(",", out);//Agent count
    fputs(",,,,,,,,", out);//Neighbourhood stats
    fputs(",,,,,,,,,", out);//Model Args
    fputs("\n", out);
    //Row 2
    for (const auto &bin : BINARIES)
    {//9 columns per model
        fputs("init (s)", out);
        fputs(",,,,,", out);
        fputs("step avg (s)", out);
        fputs(",,,", out);
        fputs("overall (s)", out);
        fputs(",", out);
    }
    fputs(",", out);//Agent count
    fputs("Neighbourhood Stats", out);
    fputs(",,,,,,,,", out);
    fputs("Model", out);
    fputs(",,,,,,,,,", out);
    fputs("\n", out);
    //Row 3
    for (const auto &bin : BINARIES)
    {//9 columns per model
        //Init
        fputs("overall", out);
        fputs(",", out);
        fputs("initCurand", out);
        fputs(",", out);
        fputs("kernel", out);
        fputs(",", out);
        fputs("pbm", out);
        fputs(",", out);
        fputs("freeCurand", out);
        fputs(",", out);
        //Step avg
        fputs("overall", out);
        fputs(",", out);
        fputs("kernel", out);
        fputs(",", out);
        fputs("texture", out);
        fputs(",", out);
        //Total
        fputs("time", out);
        fputs(",", out);
    }
    fputs("Agent Count,", out);
    //Neighbourhood stats
    fputs("First Min", out);
    fputs(",", out);
    fputs("First Max", out);
    fputs(",", out);
    fputs("First Avg", out);
    fputs(",", out);
    fputs("First SD", out);
    fputs(",", out);
    fputs("Last Min", out);
    fputs(",", out);
    fputs("Last Max", out);
    fputs(",", out);
    fputs("Last Avg", out);
    fputs(",", out);
    fputs("Last SD", out);
    fputs(",", out);
    fflush(out);
    //ModelArgs
    fputs("agentsPerCluster", out);
    fputs(",", out);
    fputs("envWidth", out);
    fputs(",", out);
    fputs("interactionRad", out);
    fputs(",", out);
    fputs("clusterCount", out);
    fputs(",", out);
    fputs("clusterRad", out);
    fputs(",", out);
    fputs("uniformDensity", out);
    fputs(",", out);
    fputs("iterations", out);
    fputs(",", out);
    fputs("seed", out);
    fputs(",", out);
    fputs("\n", out);
    fflush(out);
}

//Exec string
void ParamSet::execString(const char* executable, CirclesParams modelArgs, char **rtn)
{
    std::string buffer("\"");
    buffer = buffer.append(BIN_DIR);
    buffer = buffer.append(executable);
    buffer = buffer.append("\"");
    buffer = buffer.append(" ");
    buffer = buffer.append("-pipe");
    buffer = buffer.append(" ");
    buffer = buffer.append("-device");
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(DEVICE_ID));
    buffer = buffer.append(" ");
    buffer = buffer.append("-circles");
    buffer = buffer.append(" ");
    //buffer = buffer.append(std::to_string(modelArgs.width));
    buffer = buffer.append(std::to_string(modelArgs.agents));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.density));
    buffer = buffer.append(" ");
    //buffer = buffer.append(std::to_string(modelArgs.interactionRad));
    //buffer = buffer.append(" ");
    //buffer = buffer.append(std::to_string(modelArgs.attractionForce));
    //buffer = buffer.append(" ");
    //buffer = buffer.append(std::to_string(modelArgs.repulsionForce));
    buffer = buffer.append(std::to_string(modelArgs.forceModifier));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.iterations));
    if (modelArgs.seed != 12)
    {

        buffer = buffer.append(" ");
        buffer = buffer.append("-seed");
        buffer = buffer.append(" ");
        buffer = buffer.append(std::to_string(modelArgs.seed));
    }
    const char *src = buffer.c_str();
    *rtn = (char *)malloc(sizeof(char)*(buffer.length() + 1));
    memcpy(*rtn, src, sizeof(char)*(buffer.length() + 1));
}
void ParamSet::execString(const char* executable, NullParams modelArgs, char **rtn)
{
    std::string buffer("\"");
    buffer = buffer.append(BIN_DIR);
    buffer = buffer.append(executable);
    buffer = buffer.append("\"");
    buffer = buffer.append(" ");
    buffer = buffer.append("-pipe");
    buffer = buffer.append(" ");
    buffer = buffer.append("-device");
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(DEVICE_ID));
    buffer = buffer.append(" ");
    buffer = buffer.append("-null");
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.agents));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.density));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.iterations));
    if (modelArgs.seed != 12)
    {

        buffer = buffer.append(" ");
        buffer = buffer.append("-seed");
        buffer = buffer.append(" ");
        buffer = buffer.append(std::to_string(modelArgs.seed));
    }
    const char *src = buffer.c_str();
    *rtn = (char *)malloc(sizeof(char)*(buffer.length() + 1));
    memcpy(*rtn, src, sizeof(char)*(buffer.length() + 1));
}
void ParamSet::execString(const char* executable, DensityParams modelArgs, char **rtn)
{
    std::string buffer("\"");
    buffer = buffer.append(BIN_DIR);
    buffer = buffer.append(executable);
    buffer = buffer.append("\"");
    buffer = buffer.append(" ");
    buffer = buffer.append("-pipe");
    buffer = buffer.append(" ");
    buffer = buffer.append("-device");
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(DEVICE_ID));
    buffer = buffer.append(" ");
    buffer = buffer.append("-density");
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.agentsPerCluster));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.envWidth));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.clusterCount));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.clusterRad));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.interactionRad));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.uniformDensity));
    buffer = buffer.append(" ");
    buffer = buffer.append(std::to_string(modelArgs.iterations));
    if (modelArgs.seed != 12)
    {

        buffer = buffer.append(" ");
        buffer = buffer.append("-seed");
        buffer = buffer.append(" ");
        buffer = buffer.append(std::to_string(modelArgs.seed));
    }
    const char *src = buffer.c_str();
    *rtn = (char *)malloc(sizeof(char)*(buffer.length() + 1));
    memcpy(*rtn, src, sizeof(char)*(buffer.length() + 1));
}
std::shared_ptr<ModelParams> ParamSet::interpolateParams(std::shared_ptr<ModelParams> start, std::shared_ptr<ModelParams> end, const unsigned int step, const unsigned int totalSteps)
{
    switch (start->enumerator())
    {
    case Null:
    {
        std::shared_ptr<NullParams> a = std::make_shared<NullParams>();
        std::shared_ptr<const NullParams> s = std::dynamic_pointer_cast<const NullParams>(start);
        std::shared_ptr<const NullParams> e = std::dynamic_pointer_cast<const NullParams>(end);
        a->operator=(::interpolateParams(*s, *e, step, totalSteps));
        return a;
    }
        break;
    case Circles:
    {
        std::shared_ptr<CirclesParams> a = std::make_shared<CirclesParams>();
        std::shared_ptr<const CirclesParams> s = std::dynamic_pointer_cast<const CirclesParams>(start);
        std::shared_ptr<const CirclesParams> e = std::dynamic_pointer_cast<const CirclesParams>(end);
        a->operator=(::interpolateParams(*s, *e, step, totalSteps));
        return a;
    }
        break;
    case Density:
    {
        std::shared_ptr<DensityParams> a = std::make_shared<DensityParams>();
        std::shared_ptr<const DensityParams> s = std::dynamic_pointer_cast<const DensityParams>(start);
        std::shared_ptr<const DensityParams> e = std::dynamic_pointer_cast<const DensityParams>(end);
        a->operator=(::interpolateParams(*s, *e, step, totalSteps));
        return a;
    }
        break;
    default: break;
    }
    return nullptr;
}
std::shared_ptr<ModelParams> ParamSet::interpolateParams2D(std::shared_ptr<ModelParams> start, std::shared_ptr<ModelParams> end1, std::shared_ptr<ModelParams> end2, const unsigned int step1, const unsigned int totalSteps1, const unsigned int step2, const unsigned int totalSteps2)
{
    switch (start->enumerator())
    {
    case Null:
    {
        std::shared_ptr<NullParams> a = std::make_shared<NullParams>();
        std::shared_ptr<const NullParams> s = std::dynamic_pointer_cast<const NullParams>(start);
        std::shared_ptr<const NullParams> e1 = std::dynamic_pointer_cast<const NullParams>(end1);
        std::shared_ptr<const NullParams> e2 = std::dynamic_pointer_cast<const NullParams>(end2);
        a->operator=(::interpolateParams2D(*s, *e1, *e2, step1, totalSteps1, step2, totalSteps2));
        return a;
    }
        break;
    case Circles:
    {
        std::shared_ptr<CirclesParams> a = std::make_shared<CirclesParams>();
        std::shared_ptr<const CirclesParams> s = std::dynamic_pointer_cast<const CirclesParams>(start);
        std::shared_ptr<const CirclesParams> e1 = std::dynamic_pointer_cast<const CirclesParams>(end1);
        std::shared_ptr<const CirclesParams> e2 = std::dynamic_pointer_cast<const CirclesParams>(end2);
        a->operator=(::interpolateParams2D(*s, *e1, *e2, step1, totalSteps1, step2, totalSteps2));
        return a;
    }
        break;
    case Density:
    {
        std::shared_ptr<DensityParams> a = std::make_shared<DensityParams>();
        std::shared_ptr<const DensityParams> s = std::dynamic_pointer_cast<const DensityParams>(start);
        std::shared_ptr<const DensityParams> e1 = std::dynamic_pointer_cast<const DensityParams>(end1);
        std::shared_ptr<const DensityParams> e2 = std::dynamic_pointer_cast<const DensityParams>(end2);
        a->operator=(::interpolateParams2D(*s, *e1, *e2, step1, totalSteps1, step2, totalSteps2));
        return a;
    }
        break;
    default: break;
    }
    return nullptr;
}