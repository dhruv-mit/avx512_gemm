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



            for(int ki = 0;ki < k;ki += 32)
            {
                __m512i a = _mm512_loadu_si512((void*)&Ar[ki]);

                // uint16_t A_dump[32];
                // _mm512_storeu_si512(A_dump, a);

                // if (row == 0 && col == 0 && ki == 0) {
                //     printf("A bf16:\n");
                //     for (int t = 0; t < 32; t++)
                //         printf("%04x ", A_dump[t]);
                //     printf("\n");
                // }




                __m256i b_packed_256 = _mm256_loadu_si256((__m256i*)&Bc[ki/2]);

                // uint8_t B_dump[32];
                // _mm256_storeu_si256((__m256i*)B_dump, b_packed_256);

                // if (row == 0 && col == 0 && ki == 0) {
                //     printf("B packed bytes:\n");
                //     for (int t = 0; t < 32; t++)
                //         printf("%02x ", B_dump[t]);
                //     printf("\n");
                // }

                __m512i b_expanded = _mm512_cvtepu8_epi16(b_packed_256);

                // uint16_t B16_dump[32];
                // _mm512_storeu_si512(B16_dump, b_expanded);

                // if (row == 0 && col == 0 && ki == 0) {
                //     printf("B expanded u8->u16:\n");
                //     for (int t = 0; t < 32; t++)
                //         printf("%04x ", B16_dump[t]);
                //     printf("\n");
                // }




                // __m512i even_bf16 = _mm512_slli_epi16(
                //     _mm512_and_si512(b_expanded, _mm512_set1_epi16(0x000F)), 
                //     8
                // );

                // __m512i odd_bf16 = _mm512_slli_epi16(
                //     _mm512_and_si512(b_expanded, _mm512_set1_epi16(0x00F0)), 
                //     4
                // );

                // __mmask32 odd_mask = 0xAAAAAAAA; 
                // __m512i b_bf16 = _mm512_mask_blend_epi16(odd_mask, even_bf16, odd_bf16);


                // __m512i lo = _mm512_and_si512(b_expanded, _mm512_set1_epi16(0x000F));
                // __m512i hi = _mm512_srli_epi16(b_expanded, 4);

                                
                uint16_t tmp[32];
                _mm512_storeu_si512(tmp, b_expanded);

                for (int t = 0; t < 32; ++t) {
                    uint8_t u4 = (t & 1) ? (tmp[t] >> 4) : (tmp[t] & 0xF);
                    tmp[t] = lut[u4];
                }

                __m512i b_bf16 = _mm512_loadu_si512(tmp);



                // __m512 acc = _mm512_setzero_ps();

                __m512 acc = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh)a, (__m512bh)b_bf16);


                float reduced = _mm512_reduce_add_ps(acc);

                // printf("reduced %f\n", reduced);

                uint32_t s_bits = ((uint32_t)Sc[ki/32]) << 16;
                float scale_f = *(float*)&s_bits;

                Cr[col] += reduced * scale_f;

            }



        }
    }

}
