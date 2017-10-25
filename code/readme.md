# Spatial Partitioning Bench

The projects within this directory provide various modifications to a standard GPU implementation of spatial partitioning. This algorithm is used for surveying data within a spatial region. This is used within many agent-based models and has much overlap with the k-nearest neighbours algorithm often used in machine learning (albeit with different performance priorities).

Each modification utilises a basic configuration file `config.cu`, this provides one or more preprocessor macros, before including the rest of the code. Additionally there may be a header with mod specific functions.

It as structured in this manner to avoid having to maintain multiple copies of the same core algorithm. Storing the differences in runtime linked dlls was also considered, however dynamic linked device code is harmful to compiler optimisation, hence the choice to run with distinct executables.

Each modification can be built in command line or OpenGL mode, the latter provides a real-time visualisation of the locations stored within the data-structure. The colours of the particles move from green to red as the size of their neighbourhood increases (relative to the size of the total population).

## Included Modifications
* Default: This version is our original implementation, agents access bins individually in a linear order.
* Strips: This version merges contiguous bins into strips, reducing the total computation of switching bins by two thirds.
* Hilbert: This version stores bins according to the Hilbert space filling curve. Due to the nature of the space filling curve, it's values are encoded via a look-up table.
* Peano: This version stores bins according to the Peano space filling curve. Due to the nature of the space filling curve, it's values are encoded via a look-up table.
* MortonCompute: This version stores bins according to the Morton code (Z-order curve). This space filling technique is calculated with bit-splicing, and therefore can be computed on the fly.
* Morton: This version stores bins according to the Morton code (Z-order curve). For continuity with other space filling curves, we have also provided a look-up table version.

##Included Benchmark Models

### Circles

This model is an approximate representation of an agent-based particle model. Particles naturally move into hollow clusters until a steady state is reached.

`<executable> -circles <width> <density> <interaction radius> <attract force> <repel force> <iterations> -seed <ulong>`

Using a seed of 0 will provide uniform initialisation


### Null

This model is static, that particles never move and each iteration perform the same calculation. The intention of this is to allow better isolation of properties which are expected to affect performance.

`<executable> -null <agent count> <density> <interaction radius> <iterations> -seed <ulong>`

Using a seed of 0 will provide uniform initialisation

### Density

This is a duplication of the above Null model, however it instead uses a clustering initialisation scheme. It is understood that non-uniform distributions are likely harmful to performance, therefore we shall use this benchmark to assess such circumstances.

`<executable> -density <agent count> <environment width> <interaction radius> <cluster count> <cluster radius> iterations> -seed <ulong>`

At current a seed of 0 will not provide uniform cluster distribution.