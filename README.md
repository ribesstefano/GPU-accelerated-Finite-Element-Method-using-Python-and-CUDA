# GPU Computing with Python - FEM

This project aims at implementing a FEM solver that leverages GPU computations.

## Notes on Code

* capital letters: notation for "global" matrixes
* lower case letters: "per node" matrixes
* `e`: is notation for "element", which is another way to refer to a cell


We want to compute: `K @ a = f`, where:

* `K`: matrix representing the stiffness
* `a`: vector containing a list of DoF
* `f`: vector describing the applied force

How is `K` calculated? The matrix `K` is...