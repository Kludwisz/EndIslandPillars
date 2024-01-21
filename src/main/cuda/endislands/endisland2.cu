
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GPUJRand.h"

#include <chrono>
#include <math.h>
#include <stdio.h>
#include <inttypes.h>

#define PRINT_TIME 0
#define CUDA_CORES 3840
#define SEEDS_PER_CUDA_JOB 16
#define SEEDS_PER_BATCH CUDA_CORES * SEEDS_PER_CUDA_JOB


__device__ int processTopIslandSeed(uint64_t* seedPtr) {
    // find top end island seeds that fill the entire y-range subchunk

    // position 1
    int x1 = nextInt(seedPtr, 16);
    int y1 = nextInt(seedPtr, 16); //+ 55;
    int z1 = nextInt(seedPtr, 16);

    if (nextInt(seedPtr, 4) != 0)
        return 0;

    // position 2
    int x2 = nextInt(seedPtr, 16);
    int y2 = nextInt(seedPtr, 16); //+ 55;
    int z2 = nextInt(seedPtr, 16);

    if (x1 != x2 || z1 != z2) // only allow islands that are at the same x,z coord
        return 0;
    if (y1 != 15 && y2 != 15) // top island variant only
        return 0;

    // only allow corner islands
    if (x1 != 0 && x1 != 15) return 0;
    if (z1 != 0 && z1 != 15) return 0;

    // island 1
    int height1 = 0;
    double r = nextInt(seedPtr, 3) + 4.0;
    if (r < 5.5) return 0; // optimization feature

    while (r > 0.5) {
        r -= nextInt(seedPtr, 2) + 0.5;
        height1++;
    }

    // island 2
    int height2 = 0;
    r = nextInt(seedPtr, 3) + 4.0;
    if (r < 5.5) return 0; // optimization feature

    while (r > 0.5) {
        r -= nextInt(seedPtr, 2) + 0.5;
        height2++;
    }
    
    // check if the islands combined cover the entire 16-block y span
    int topY, bottomY;
    int topH, bottomH;

    if (y1 > y2) {
        topY = y1 + 55;  
        bottomY = y2 + 55;
        topH = height1; 
        bottomH = height2;
    }
    else {
        topY = y2 + 55;
        bottomY = y1 + 55;
        topH = height2;
        bottomH = height1;
    }

    if (topY - topH > bottomY) // islands don't overlap
        return 0;
    if (topY - (bottomY - bottomH) < 18) // total height span is too small
        return 0;

    // found a good top island seed
    printf("TOP %d %d ", x1, z1);
    return 1; 
}


__device__ int processBottomIslandSeed(uint64_t* seedPtr) {
    // find top end island seeds that fill the entire y-range subchunk

    // position 1
    int x1 = nextInt(seedPtr, 16);
    int y1 = nextInt(seedPtr, 16); //+ 55;
    int z1 = nextInt(seedPtr, 16);

    if (nextInt(seedPtr, 4) != 0)
        return 0;

    // position 2
    int x2 = nextInt(seedPtr, 16);
    int y2 = nextInt(seedPtr, 16); //+ 55;
    int z2 = nextInt(seedPtr, 16);

    if (x1 != x2 || z1 != z2) // only allow islands that are at the same x,z coord
        return 0;
    if (y1 != 0 && y2 != 0) // bottom island variant only
        return 0;

    // only allow corner islands
    if (x1 != 0 && x1 != 15) return 0;
    if (z1 != 0 && z1 != 15) return 0;

    // island 1
    int height1 = 0;
    double r = nextInt(seedPtr, 3) + 4.0;
    if (r < 5.5) return 0; // optimization feature

    while (r > 0.5) {
        r -= nextInt(seedPtr, 2) + 0.5;
        height1++;
    }

    // island 2
    int height2 = 0;
    r = nextInt(seedPtr, 3) + 4.0;
    if (r < 5.5) return 0; // optimization feature

    while (r > 0.5) {
        r -= nextInt(seedPtr, 2) + 0.5;
        height2++;
    }

    // check if the bottom island has the maximum possible height
    if (y1 == 0) {
        if (height1 < 11 || height2 < 7) // max height for island 1, good height for 2
            return 0;
    }
    if (y2 == 0) {
        if (height2 < 11 || height1 < 7) // max height for island 2, good height for 1
            return 0;
    }

    // found a good top island seed
    printf("BOT %d %d ", x1, z1);
    return 1;
}


extern "C" __global__ void findChunkSeeds(uint64_t batchNr) {
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

                // if (processTopIslandSeed(seedPtr)) {
                if (processBottomIslandSeed(seedPtr)) {
                    *seedPtr = high31 | low17;
                    goBack(seedPtr);
                    *seedPtr ^= 0x5deece66dULL;

                    printf("%" PRIu64 "\n", *seedPtr);
                }

                // > C:\Users\kludw\source\repos\DesertTempleNotch\DesertTempleNotch\popseeds.txt
            }
        }
    }
}

// ETA:  ~10 minutes
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
        if (PRINT_TIME && batch % 100 == 0) {
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = end - start;
            long long nanoTime = elapsed.count();
            long long seconds = nanoTime / 1000'000'000;
            printf("=== Took %lld seconds\n", seconds);
            printf("=== Batch %d / %lld\n", batch, batches);
        }

        findChunkSeeds <<< 30, 128 >>> (batch);

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