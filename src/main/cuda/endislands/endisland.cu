
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GPUJRand.h"

#include <chrono>
#include <math.h>
#include <stdio.h>
#include <inttypes.h>

#define CUDA_CORES 3840
#define SEEDS_PER_CUDA_JOB 16
#define SEEDS_PER_BATCH CUDA_CORES * SEEDS_PER_CUDA_JOB


extern "C" __global__ void run(uint64_t batchNr) {
    int indexInBatch = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t seedsPerCudaJob = SEEDS_PER_CUDA_JOB;
    const uint64_t seedsPerBatch = SEEDS_PER_BATCH;

    uint64_t seed = 0;
    uint64_t* seedPtr = &seed;

    if (indexInBatch < seedsPerBatch / seedsPerCudaJob) {
        for (int seedIndex = 0; seedIndex < seedsPerCudaJob; seedIndex++) {

            uint64_t high31 = ((batchNr * seedsPerBatch + indexInBatch * seedsPerCudaJob + seedIndex) * 14LL) << 17;

            for (uint64_t low17 = 0; low17 < 131072LL; low17++) {
                *seedPtr = high31 | low17; // setSeedFromIntUnscrambled

                // -------------------------------------------------------------------------
                // run all the necessary rand calls here - return on encountered discrepancy
                // -------------------------------------------------------------------------

                // position 1
                int x1 = nextInt(seedPtr, 16);
                int y1 = nextInt(seedPtr, 16) + 55;
                int z1 = nextInt(seedPtr, 16);

                if (nextInt(seedPtr, 4) != 0)
                    continue;

                // position 2
                int x2 = nextInt(seedPtr, 16);
                int y2 = nextInt(seedPtr, 16) + 55;
                int z2 = nextInt(seedPtr, 16);

                if (x1 != x2 || z1 != z2) // only allow islands that are at the same x,z coord
                    continue;

                // --- island shape calls


                // island 1
                int height1 = 0;
                double r = nextInt(seedPtr, 3) + 4.0;

                while (r > 0.5) {
                    r -= nextInt(seedPtr, 2) + 0.5;
                    height1++;
                }

                // island 2
                int height2 = 0;
                r = nextInt(seedPtr, 3) + 4.0;

                while (r > 0.5) {
                    r -= nextInt(seedPtr, 2) + 0.5;
                    height2++;
                }

                //if (height1 + height2 > 16)
                //    printf("H%d\n", height1 + height2);

                if (height1 + height2 < 22)
                    continue;

                *seedPtr = high31 | low17;
                goBack(seedPtr);
                printf("%d %" PRId64 "\n", height1 + height2, *seedPtr);
            }
        }
    }
}

// ETA:  ~8 minutes
int main()
{
    const long long totalSeeds = ceil((double)(1ULL << 31) / (double)14);
    const long long seedsPerCudaJob = SEEDS_PER_CUDA_JOB;
    const long long seedsPerBatch = SEEDS_PER_BATCH;
    const long long batches = totalSeeds / seedsPerBatch;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (long long batch = 0; batch < batches; batch++) {
        if (batch % 100 == 0) {
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = end - start;
            long long nanoTime = elapsed.count();
            long long seconds = nanoTime / 1000'000'000;
            printf("=== Took %lld seconds\n", seconds);
            printf("=== Batch %d / %lld\n", batch, batches);
        }

        run <<< 30, 128 >>> (batch);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
            goto Error;
        }

        // reset device
        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return 1;
        }
        continue;

    Error:
        return 1;
    }

    // reset device before returning
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


// bit lifting was the first idea, since island generation uses
// a lot of nextInts bounded by powers of 2
// Unfortunately, the code didn't work as expected and I finally
// decided to abandon the idea and do a full bruteforce on the GPU


// @Deprecated
__host__ void pillarHeightLifting() {
    uint64_t seed = 0;
    uint64_t* rand = &seed;

    uint64_t seed1 = 0;
    uint64_t* rand1 = &seed1;
    uint64_t seed2 = 0;
    uint64_t* rand2 = &seed2;

    // int maxheight = 0;
    const int maxheight = 22;

    for (uint64_t low18 = 0; low18 < (1LL << 18); low18++) {
        setSeedFromIntUnscrambled(rand, low18);

        nextInt(rand, 14);  // first spawns

        nextInt(rand, 16); // position of 1st island
        nextInt(rand, 16);
        nextInt(rand, 16);

        nextInt(rand, 4); // second spawns

        nextInt(rand, 16); // position of end island
        nextInt(rand, 16);
        nextInt(rand, 16);

        // island placement

        nextInt(rand, 3); // radius

        // assume each possible output of fastBoundedNextInt(3) and calculate pillar height
        for (int val1 = 0; val1 < 3; val1++) {
            setSeedFromIntUnscrambled(rand1, *rand);
            int height1 = 0;
            double r = val1 + 4.0;

            while (r > 0.5) {
                r -= nextInt(rand1, 2) + 0.5;
                height1++;
            }

            nextInt(rand1, 3); // radius

            // assume each possible output of fastBoundedNextInt(3) and calculate pillar height
            for (int val2 = 0; val2 < 3; val2++) {
                setSeedFromIntUnscrambled(rand2, *rand1);

                int height2 = 0;
                r = val2 + 4.0;

                while (r > 0.5) {
                    r -= nextInt(rand2, 2) + 0.5;
                    height2++;
                }

                if (height1 + height2 >= maxheight) {
                    printSeed(low18);
                    //printf("%d\n", height1 + height2);
                    //maxheight = height1 + height2;
                }
            }
        }
    }

    //printf("Max height of End Island:  %d\n", maxheight);
}

// @Deprecated
extern "C" __global__ void runLifted(uint64_t low18) {
    const uint64_t seedsPerWorker = (1LL << 30) / CUDA_CORES;
    //printSeed(seedsPerWorker);

    int workerIndex = blockIdx.x * blockDim.x + threadIdx.x;
    
    uint64_t startSeed = workerIndex * seedsPerWorker;
    uint64_t endSeed = (workerIndex + 1) * seedsPerWorker;
    uint64_t populationSeed;
    uint64_t randSeed = 0;
    uint64_t* rand = &randSeed;

    // printf("%d\n", workerIndex);

    for (uint64_t high30 = startSeed; high30 <= endSeed; high30++) {
        populationSeed = (high30 << 18) | low18;

        setSeedFromIntUnscrambled(rand, populationSeed);
        
        if (nextInt(rand, 14) != 0)
            continue;

        // position 1
        int x1 = nextInt(rand, 16);
        int y1 = nextInt(rand, 16) + 55;
        int z1 = nextInt(rand, 16);

        if (nextInt(rand, 4) != 0)
            continue;

        // position 2
        int x2 = nextInt(rand, 16);
        int y2 = nextInt(rand, 16) + 55;
        int z2 = nextInt(rand, 16);

        if (x1 != x2 || z1 != z2) // only allow islands that are at the same x,z coord
            continue;

        // --- island shape calls
        

        // island 1
        int height1 = 0;
        double r = nextInt(rand, 3) + 4.0;

        while (r > 0.5) {
            r -= nextInt(rand, 2) + 0.5;
            height1++;
        }

        // island 2
        int height2 = 0;
        r = nextInt(rand, 3) + 4.0;

        while (r > 0.5) {
            r -= nextInt(rand, 2) + 0.5;
            height2++;
        }

        //if (height1 + height2 > 16)
        //    printf("H%d\n", height1 + height2);

        if (height1 + height2 < 20)
            continue;

        // for now, don't check position - just print the seed
        printf("%" PRId64 "   %d\n", populationSeed, height1 + height2);
    }
}

// @Deprecated
__host__ int launchBitLiftedFinderKernel(uint64_t low18) {
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    runLifted <<< 30, 128 >>> (low18);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
        goto Error;
    }

    // reset device before returning
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        goto Error;
    }
    return 0;

    Error:
        return 1;
}

// @Deprecated
int main2() {
    // pillarHeightLifting();
    // launchBitLiftedFinderKernel(44885LL);
    
    uint64_t low18s[] = {
        13667,
        14900,
        31251,
        31389,
        36686,
        42619,
        62305,
        71745,
        80027,
        82226,
        94105,
        104803,
        104803,
        167174,
        172078,
        179496,
        198584,
        201658,
        209306,
        238857,
        242951,
        246141,
        253894,
        260656
    };

    int seedcount = 24;

    for (int i = 0; i < seedcount; i++) {
        launchBitLiftedFinderKernel(low18s[i]);
    }

    return 0;
}