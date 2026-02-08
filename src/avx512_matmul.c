#include "avx512_kernel.h"

#pragma once

void matmul(uint16_t* A, uint8_t* B, uint16_t* S, float* C, int M, int N, int K, const uint16_t* lut)
{
    int mr;
    int nr;


    for(int row = 0;row < M;row += mr)
    {
        for(int col = 0;col< N;col += nr)
        {
            kernel_bf16_int4_bf16(A,B,S,C,row,col, K, lut);
        }
    }
}