#include "avx512_kernel.h"

void kernel_bf16_int4_bf16(
    const uint16_t* A,             //A matrix
    const uint8_t * B,              //B matrix(2 elements per 8 bit)
    const uint16_t* S,               //sclae matrix
    float* C,                        //C matrix
    int r,
    int c,
    int k,
    const uint16_t* lut
)
{

    for(int row = 0;row < r;row++)
    {
        const uint16_t* Ar = A + row*k;

        float* Cr = C + row*c;

        for(int col = 0; col < c;col++)
        {
            const uint8_t* Bc = B + (col * k)/2;
            const uint16_t* Sc = S + (col * k)/32;

            __m512 acc = _mm512_setzero_ps();


            for(int ki = 0;ki < k;ki += 32)
            {
                __m512i a = _mm512_loadu_si512((void*)&Ar[ki]);



                __m256i b_packed_256 = _mm256_castsi128_si256(
                    _mm_loadu_si128((__m128i*)&Bc[ki/2])
                );
                __m512i b_expanded = _mm512_cvtepu8_epi16(b_packed_256);


                __m512i even_bf16 = _mm512_slli_epi16(
                    _mm512_and_si512(b_expanded, _mm512_set1_epi16(0x000F)), 
                    8
                );

                __m512i odd_bf16 = _mm512_slli_epi16(
                    _mm512_and_si512(b_expanded, _mm512_set1_epi16(0x00F0)), 
                    4
                );


                __mmask32 odd_mask = 0xAAAAAAAA; 
                __m512i b_bf16 = _mm512_mask_blend_epi16(odd_mask, even_bf16, odd_bf16);


                acc = _mm512_dpbf16_ps(acc, (__m512bh)a, (__m512bh)b_bf16);

            }

            float reduced = _mm512_reduce_add_ps(acc);

            uint32_t s_bits = ((uint32_t)*Sc) << 16;
            float scale_f = *(float*)&s_bits;

            Cr[col] += reduced * scale_f;

        }
    }

}
