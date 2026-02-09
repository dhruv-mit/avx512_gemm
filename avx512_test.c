#include <stdio.h>
#include <stdlib.h>
#include<stdint.h>
#include <math.h>
#include "src/avx512_kernel.h"


static const uint16_t lut[16] = {
    0x0000, // 0
    0x3f80, // 1
    0x4000, // 2
    0x4040, // 3
    0x4080, // 4
    0x40a0, // 5
    0x40c0, // 6
    0x40e0, // 7
    0x4100, // 8
    0x4110, // 9
    0x4120, // 10
    0x4130, // 11
    0x4140, // 12
    0x4150, // 13
    0x4160, // 14
    0x4170  // 15
};

static inline float bf16_to_fp32(uint16_t x) {
    union {
        uint32_t u;
        float f;
    } tmp;

    tmp.u = ((uint32_t)x) << 16;
    return tmp.f;
}

int main() {

    const int r = 200;
    const int c = 200;
    const int k = 6400;   // MUST be multiple of 32
    const int mr = 2;   // should divide r for now
    const int nr = 1;   // should divide c for now

    uint16_t A[r*k];
    uint8_t  B[c*k/2];
    uint16_t S[c*(k/32)];
    float C_ref[r*c];
    float C[r*c];

    // ---- random A
    for (int i = 0; i < r*k; i++) {
        float x = ((rand() % 2000) - 1000) / 1000.0f;
        union { float f; uint32_t u; } t;
        t.f = x;
        A[i] = t.u >> 16;
        // A[i] = 0x3f80;
    }
    // ---- random B (0..15)
    for (int i = 0; i < c*k/2; i++)
        B[i] = rand() & 0xFF;
        // B[i] = 0x11;

    // ---- random scales
    for (int i = 0; i < c*(k/32); i++) {
        float s = ((rand() % 2000) / 1000.0f);
        union { float f; uint32_t u; } t;
        t.f = s;
        S[i] = t.u >> 16;
        // S[i] = 0x3f80;
    }

    for (int i = 0; i < r*c; ++i)
        C[i] = 0.0f;     
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
                lut,
                c
            );
        }
    }



    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {

            float acc = 0.0f;

            for (int kk = 0; kk < k; ++kk) {

                // ---- A (bf16) ----
                float a = bf16_to_fp32(A[i*k + kk]);

                // ---- unpack unsigned int4 ----
                int byte_idx = j*(k/2) + kk/2;
                uint8_t packed = B[byte_idx];

                uint8_t u4;
                if ((kk & 1) == 0)
                    u4 = packed & 0x0F;
                else
                    u4 = (packed >> 4) & 0x0F;

                float b = (float)u4;

                // ---- scale for group of 32 ----
                int scale_idx = j*(k/32) + kk/32;
                float scale = bf16_to_fp32(S[scale_idx]);

                acc += a * (b * scale);
            }

            C_ref[i*c + j] = acc;
        }
    }


    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int bad_count = 0;

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {

            int idx = i*c + j;

            float v = C[idx];
            float ref = C_ref[idx];

            float abs_err = fabsf(v - ref);
            float rel_err = abs_err / (fabsf(ref) + 1e-6f);

            if (abs_err > max_abs_err)
                max_abs_err = abs_err;

            if (rel_err > max_rel_err)
                max_rel_err = rel_err;

            // print first few mismatches
            if (abs_err > 1e-3f && bad_count < 10) {
                printf("Mismatch at (%d,%d): C=%f  C_ref=%f  abs=%g rel=%g\n",
                    i, j, v, ref, abs_err, rel_err);
                bad_count++;
            }

            // printf("(%d,%d): C=%f  C_ref=%f  diff=%f\n",i,j,v,ref,v-ref);
        }
    }

    printf("Max abs error = %g\n", max_abs_err);
    printf("Max rel error = %g\n", max_rel_err);


    return 0;
}
