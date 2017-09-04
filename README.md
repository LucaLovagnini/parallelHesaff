# Parallel Hessian Affine (PHA) Detector with SIFT Descriptor

OpenMP parallelization of  the Hessian Affine detector with SIFT descriptor (original code [here](http://github.com/perdoch/hesaff) ). Important details about this project can be found in the report file in this repository.

## Tested Architectures

This project has been tested on:

- Intel®Xeon Phi TM 7210, aka Knights Landing (KNL), 64 cores, with NUMA memory architecture.
- Intel(R) Core(TM) i7-5820K, 6 cores, DD3 Memory, 1536 kB L2 Cache, 15360 kB L3 Cache.

## Compilers and Libraries

- C++ Intel Compiler 2017.3.191.
- OpenCV 3.2 commit 9053839
- OpenCV contrib commit f39f415

## User Manual

    ./hessaff image [maxThreads][nRuns][scability]

Where:

- `image` is the input image. The image extension must be supported by OpenCV.
- `maxThreads` (default: omp_get_max_threads()) is the upper bound of the number of threads
used in the program execution. Look at scalability for more details.
- `nRuns` (defualt: 1) how many times each parallel configuration is executed. This is useful for
evaluation purpose.
- `scalability` (default: 0) if 1 perform a scalability test, otherwise get the descriptors in image
using maxThreads threads for nRuns times. In a scalability test, maxThreads different parallel
configurations are executed: the first one uses omp_set_num_threads(1), the i-th configura-
tion uses omp_set_num_threads(i) and the last one omp_set_num_threads(maxThreads).
Each configuration is executed nRuns times.

A typical execution output is:

    [spm1428@ninja hesaffCC]$ ./hesaff darkKnight.jpg
    ---------------------------------------------------
    File: darkKnight.jpg
    Using 64 threads
    descriptor time 2.13762 seconds
    parallel time=1.00586
    descriptors.rows=41131
    avg descriptor time=2.13762
    avg parallel time=1.00586

## Performance

### Datasets:

- Movie Posters
- Paintings images
- [Oxford Building Dataset](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)

### KNL Scalability (from 400 to 6000 pxs)

![alt text](https://github.com/lovaj/parallelHesaff/blob/master/Figures/knlScalability.png)

### Timings on Intel i7-5820K

![](https://github.com/lovaj/parallelHesaff/blob/master/Figures/i7times.png)

### i7-5820K Speedup (over original code)

![](https://github.com/lovaj/parallelHesaff/blob/master/Figures/i7speedup.png)


