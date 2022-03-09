
An illustration of simple OpenMP offload reduction operation over multiple ranks and GPUs

Prerequisite modules:
module load rocm openmpi

To compile and run:
```
make
mpiexec -n <nprocs> ./norm
```

For help:
```
make help
```
