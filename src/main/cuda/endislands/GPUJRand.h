#pragma once

#include "stdint.h"
#include "inttypes.h"
#include "stdio.h"

#define CPU_OR_GPU __device__ __host__
#define MASK48 ((1ULL << 48u) - 1ULL)

// ------------------------------------------------------------------
// Credits to Cubitect for the cubiomes library, I used parts of the
// Java Random implementation here
// https://github.com/Cubitect/cubiomes
// ------------------------------------------------------------------

CPU_OR_GPU inline void printSeed(uint64_t seed) {
    printf("%" PRIu64 "\n", seed);
}

CPU_OR_GPU inline int32_t Next(uint64_t* seed, int bits)
{
    *seed = (*seed * 0x5deece66du + 0xBu) & MASK48;

    return (int32_t)(*seed >> (48u - bits));
}

CPU_OR_GPU inline void advance(uint64_t* seed)
{
    *seed = (*seed * 0x5deece66du + 0xBu) & MASK48;
}

CPU_OR_GPU inline void advance2(uint64_t* seed)
{
    *seed = (*seed * 205749139540585ULL + 277363943098ULL) & MASK48;
}

CPU_OR_GPU inline void advance3(uint64_t* seed)
{
    *seed = (*seed * 233752471717045ULL + 11718085204285ULL) & MASK48;
}

CPU_OR_GPU inline void goBack(uint64_t* seed)
{
    *seed = (*seed * 246154705703781ULL + 107048004364969ULL) & MASK48;
}

CPU_OR_GPU inline void goBack2(uint64_t* seed)
{
    *seed = (*seed * 254681119335897ULL + 120305458776662ULL) & MASK48;
}

CPU_OR_GPU inline void setSeedFromInt(uint64_t* seed, uint64_t from) {
    *seed = (uint64_t)(from ^ 0x5deece66dULL) & MASK48;
}

CPU_OR_GPU inline void setSeedFromIntUnscrambled(uint64_t* seed, uint64_t from) {
    *seed = (uint64_t)(from) & MASK48;
}

CPU_OR_GPU inline int nextInt(uint64_t* seed, const int n)
{
    int bits, val;
    const int m = n - 1;

    if ((m & n) == 0) {
        uint64_t x = n * (uint64_t)Next(seed, 31);
        return (int)((int64_t)x >> 31);
    }

    do {
        bits = Next(seed, 31);
        val = bits % n;
    } while (bits - val + m < 0);
    return val;
}

CPU_OR_GPU inline void fastBoundedNextInt(uint64_t* seed, int n, int* valPtr)
{
    *valPtr = Next(seed, 31) % (int32_t)n;
}

CPU_OR_GPU inline int32_t fastBoundedNextInt(uint64_t* seed, int n)
{
    return Next(seed, 31) % (int32_t)n;
}

CPU_OR_GPU inline bool nextBoolean(uint64_t* seed) {
    return Next(seed, 1) != 0;
}

CPU_OR_GPU inline double nextDouble(uint64_t* seed)
{
    return (((uint64_t)Next(seed, 26) << 27u) + Next(seed, 27))
        / (double)(1ULL << 53u);
}

CPU_OR_GPU inline float nextFloat(uint64_t* seed) {
    return Next(seed, 24) / ((float)(1 << 24));
}

CPU_OR_GPU inline uint64_t nextLong(uint64_t* seed)
{
    return ((uint64_t)Next(seed, 32) << 32) + Next(seed, 32);
}
