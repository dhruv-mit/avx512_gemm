#include "avx512_kernel.h"

#pragma once

void matmul(uint16_t* A, uint8_t* B, uint16_t* S, float* C, int M, int N, int K, const uint16_t* lut)
{
    int mr = 32;
    int nr = 16;


    for (int m0 = 0; m0 < M; m0 += mr) 
    {
        for (int n0 = 0; n0 < N; n0 += nr) 
        {
            kernel_bf16_int4_bf16(
                A + m0*K,
                B + n0*(K/2),
                S + n0*(K/32),
                C + m0*N + n0,
                mr,
                nr,
                K,
                lut,
                N
            );
        }
    }
}