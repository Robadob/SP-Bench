#Validation

Due to the effects of floating point precision being compounded after multiple iterations, validation compares the results of each model after a single iteration working from a unified initial state.

##Model 1
* Width: 100
* Density: 0.01
* Interaction Radius: 5
* Attraction Force: 0.1
* Repulsion Force: 0.1
* Calculated Agent Count: 10,000

###Results
Min Difference
|          | Init State | FLAMEGPU | MASON | Repast | Personal |
|----------|------------|----------|-------|--------|----------|
| Init     | -          | -        | -     | -      | -        |
| FLAMEGPU | 0.365      | -        | -     | -      | -        |
| MASON    | 0.365      | 0.0      | -     | -      | -        |
| Repast   | 0.365      | 0.0      | 0.0   | -      | -        |
| Personal | 0.365      | 0.0      | 0.0   | 0.0    | -        |

Max Difference
|          | Init State | FLAMEGPU | MASON   | Repast  | Personal |
|----------|------------|----------|---------|---------|----------|
| Init     | -          | -        | -       | -       | -        |
| FLAMEGPU | 135.996    | -        | -       | -       | -        |
| MASON    | 135.996    | 92.689   | -       | -       | -        |
| Repast   | 136.604    | 111.281  | 111.281 | -       | -        |
| Personal | 135.996    | 0.0001   | 92.689  | 111.281 | -        |

Mean difference
|          | Init State | FLAMEGPU | MASON | Repast | Personal |
|----------|------------|----------|-------|--------|----------|
| Init     | -          | -        | -     | -      | -        |
| FLAMEGPU | 51.985     | -        | -     | -      | -        |
| MASON    | 52.010     | 0.231    | -     | -      | -        |
| Repast   | 52.035     | 1.560    | 1.329 | -      | -        |
| Personal | 51.985     | 0.0      | 0.231 | 1.560  | -        |

Agents with over 1 unit difference
|          | Init State | FLAMEGPU | MASON | Repast | Personal |
|----------|------------|----------|-------|--------|----------|
| Init     | -          | -        | -     | -      | -        |
| FLAMEGPU | 9982       | -        | -     | -      | -        |
| MASON    | 9982       | 42       | -     | -      | -        |
| Repast   | 9985       | 455      | 415   | -      | -        |
| Personal | 9982       | 0        | 42    | 455    | -        |