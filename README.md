# SP-Bench
Spatial Partitioning Benchmark Suite

##Code
The `code` directory contains the Visual Studio 2013 solution required building and executing the suite of benchmarks developed against the various spatial partitioning implementations. 

###Dependencies
* [CUDA Toolkit 7.5](https://developer.nvidia.com/cuda-toolkit)
* [GLM](http://glm.g-truc.net/)(Included in repo)

##Report
The `report` directory contains the necessary LaTeX and graphical resources (and windows batch files) to build the report as a pdf file. This will be output to the report directory as `paper.pdf`.

This has been tested using Miktex 2.9, but there are no obvious reasons that other versions shouldn't work. If there are compilation issues, it is likely that the included LaTeX packages are different versions, whereby syntax has changed.
