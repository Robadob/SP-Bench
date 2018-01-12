# Poster Description (appears on website)
## <800 char

We present techniques for the optimisation accessing the uniform spatial partitioning data structure to benefit complex systems simulations on GPUs. We have investigated how changes to bin access patterns can improve the performance achieved when accessing stored neighbour data, by reducing branch divergence and scattered memory accesses. Results collected using a Titan X Pascal show improvements as significant as 5x speedup when compared with the standard access patterns.


# Extended Abstract/Results (used for submission review only)
## <3000 char

This research presents techniques for the optimisation accessing the uniform spatial partitioning data structure to benefit complex systems simulations on GPUs.

Many complex systems have mobile entities located within a continuous space such as: particles, people or vehicles. Typically these systems are represented via agent based simulations where entities are agents. In order for these mobile agents to decide actions, they must be aware of their neighbouring agents. On GPUs this awareness is typically provided by uniform spatial partitioning, whereby each agent considers the properties of every other agent located within a spatial radial area about their simulated position.

This near neighbours process, although largely simulation independent, requires a large quantity of scattered memory reads. As such the performance of this process often has a significant impact on that of the overall simulation. Therefore improvements to the performances of accesses to uniform spatial partitioning are capable of benefiting a diverse range of simulations such as those in the fields of transport and physics.

At this stage our results have shown that high level changes to how bins are accessed within the data structure can provide improvements as significant as a 5x speed up (results within were collected using a Titan X (Pascal) GPU).

This research is ongoing and has only currently considered the impact of problem size and neighbourhood size. In the continued research we are exploring the effects of further parameters, including those which are model and hardware specific. This will allow us to provide the ability to automatically tune the uniform spatial partitioning data structure for any given application. 


# Bio
## <1000 char

Robert is a PhD student in the Visual Computing research group at the University of Sheffield. He is researching optimisations to the accessing of spatial data structures when applied to large scale complex systems simulations. He also assists in the teaching of both 3D Graphics (OpenGL) and CUDA programming.

His prior work includes applying pedestrian modelling techniques to GPU hardware, and combining distinct transport simulations to observe high-level network effects in a multi-modal transport network.

His outside interests include powerlifting, baking and developing a bespoke graphics engine.
