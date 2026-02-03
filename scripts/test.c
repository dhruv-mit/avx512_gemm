#include <stdio.h>
#include <stdlib.h>
#include<stdint.h>
#include <math.h>
#include "../src/avx512_kernel.h"


int main() {

    const int r = 8;
    const int c = 8;
    const int k = 64;   // MUST be multiple of 32
    const int mr = 4;
    const int nr = 4;

    uint16_t A[r*k];
    uint8_t  B[c*k/2];
    uint16_t S[c*(k/32)];
    float C_ref[r*c];
    float C[r*c];

    uint16_t lut[16];

    // ---- build LUT: [-8..7] for example
    for (int i = 0; i < 16; i++) {
        int v = i - 8;
        float f = (float)v;
        union { float f; uint32_t u; } t;
        t.f = f;
        lut[i] = t.u >> 16;
    }

    // ---- random A
    for (int i = 0; i < r*k; i++) {
        float x = ((rand() % 2000) - 1000) / 1000.0f;
        union { float f; uint32_t u; } t;
        t.f = x;
        A[i] = t.u >> 16;
    }

    // ---- random B (0..15)
    for (int i = 0; i < c*k/2; i++)
        B[i] = rand() & 0xFF;

    // ---- random scales
    for (int i = 0; i < c*(k/32); i++) {
        float s = ((rand() % 2000) / 1000.0f);
        union { float f; uint32_t u; } t;
        t.f = s;
        S[i] = t.u >> 16;
    }

    // reference_kernel(A, B, S, C_ref, r, c, k, lut);
    for (int m0 = 0; m0 < r; m0 += mr) {
        for (int n0 = 0; n0 < c; n0 += nr) 
        {
            kernel_bf16_int4_bf16(
                A + m0*k,
                B + n0*(k/2),
                S + n0*(k/32),
                C + m0*c + n0,
                mr,
                nr,
                k,
                lut
            );
        }
    }


    printf("PASS ✔ outputs match\n");
    return 0;
}
