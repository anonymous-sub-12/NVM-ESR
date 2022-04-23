

NVM-ESR: Using Non-Volatile Memory in Exact State
Reconstruction of Preconditioned Conjugate Gradient
============
In this work, we propose, implement and evaluate NVM-ESR, a
novel NVRAM-based mechanism for scalable and resource-efficient
exact state reconstruction for distributed linear iterative solvers. We
consider two architectures: A homogeneous cluster architecture,
in which each node stores recovery data locally, and a persistent
recovery data (PRD) sub-cluster architecture, in which all recovery
data is stored remotely on a sub-cluster of PRD nodes (in our system,
we used a single PRD node). We implemented several variants of
ESR, in which recovery data is stored in either DRAM, NVRAM, or
an SSD storage device. We conducted a comprehensive performance
evaluation of these implementations in terms of the memory and
time overheads they incur.
The results of our evaluation show that NVM-ESR is significantly
superior to ESR in terms of the size of the memory footprint required
for storing the recovery data. Furthermore, compared with NVM-
ESR in the PRD sub-cluster architecture (henceforth denoted NVM-
ESR/PRD), ESR incurs memory overhead that is larger by a factor
proportional to the product of the total number of processes and
the maximal number of simultaneous failures that the system can
recover from. NVM-ESR is also superior to ESR in terms of the
time overhead incurred by writing the recovery data, except for
small numbers of processes that can fit inside a single node in the
homogeneous architecture.
We have implemented NVM-
ESR/PRD by using MPI one-sided communication over InfiniBand??s
remote direct memory access (RDMA) towards NVRAM and have
optimized its usage by ESR??s persistence iterations. To the best of
our knowledge, our work is the first to report on a scientific applica-
tion implementation that accesses remote NVRAM in this manner.
We show how to support recoverability of such applications using
NVRAM as a substitute to traditional checkpointing.

Our implementations are build on the code supplied by Carlos Pachajoa, Wilfried N. Gansterer et al. based on their previous work
on Exact State Reconstruction with Period (ESRP) for the Conjugate Gradient solver [[1]](#1). 

**Compiling prerequisite library - MPI pmem ext:**

The MPI One-Sided Communication Pmem extension library needs to be compiled and linked to our implementation as follows:
```
cd NVMESR/mpi-pmem-ext
```
```
module load gnu/8.1.0 mpich/3.2 pmdk/1.9
``` 
```
autoreconf -i
```

supply the path to install the library (with ```--path```) and paths to mpicc, mpic++ (with ```CC``` and ```CXX```) and run:
```
./configure --prefix=YOUR_INSTALLED_MPI_PMEM_EXT_DIR CC=/opt/ohpc/pub/mpi/mpich-gnu-ohpc/3.2.1/bin/mpicc CXX=/opt/ohpc/pub/mpi/mpich-gnu-ohpc/3.2.1/bin/mpic++
``` 

then, the library will be installed with the next commands:
```
make
```
```
make install
```

**Compiling NVM-ESR implementation:**
```
cd NVMESR/esrp_branch
```
```
module load gnu/8.1.0 mpich/3.2 pmdk/1.9 gsl/2.4 openblas/0.2.20
``` 
- make sure all the relevant libraries are exported and match the linking paths in ```Makefile```.
- make sure ```libmpi-pmem-one-sided.so.0``` is on ```LD_LIBRARY_PATH```
```
make
```

**Running NVM-ESR:**
```
module load gnu/8.1.0 mpich/3.2 pmdk/1.9 gsl/2.4 openblas/0.2.20
```
- See ```NVMESR/esrp_branch/run_experiments_with_slurm.sh``` for execution examples via SLURM on your cluster.

## References
<a id="1">[1]</a> 
Pachajoa, Carlos and Pacher, Christina and Levonyak, Markus and Gansterer, Wilfried N., 
Algorithm-based checkpoint-recovery for the conjugate gradient method,
49th international conference on parallel processing-icpp, p. 1-11, 2020.

