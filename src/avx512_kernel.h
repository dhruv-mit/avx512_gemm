#include <immintrin.h>
#include <stdint.h>
#include "avx512_matmul.h"
#include <cstdio>
#include <omp.h>

void kernel_bf16_int4_bf16(
    const uint16_t* A,             //A matrix
    const uint8_t * B,              //B matrix(2 elements per 8 bit)
    const uint16_t* S,               //sclae matrix
    float* C,                        //C matrix
    // int da,                       //not taking padding  
    // int db,                       //into account
    int r,
    int c,
    int k,
    const uint16_t* lut,
    int N
);