#include "avx512_kernel.h"


void kernel_bf16_int4_bf16(
    const uint16_t* A,             //pointer to start of sub matrix of A
    const uint8_t * B,              //pointer to start of sub matrix of B(2 elements per 8 bit)
    const uint16_t* S,               //pointer to start of submatrix of sclae matrix
    float* C,                        //pointer to start of sub matrix of C
    int r,                           //number of rows in submatrix of A
    int c,                           //number of cols in submatrix of B
    int k,                           //common dim
    const uint16_t* lut,              //lookup table

    int col_size_of_c              // for testing purposes rn
)
{

    for(int row = 0;row < r;row++)
    {
        const uint16_t* Ar = A + row*k;

        float* Cr = C + row*col_size_of_c;

        for(int col = 0; col < c;col++)
        {
            const uint8_t* Bc = B + (col * k)/2;
            const uint16_t* Sc = S + (col * k)/32;



            for(int ki = 0;ki < k;ki += 32)
            {
                __m512i a = _mm512_loadu_si512((void*)&Ar[ki]);


                __m256i b_packed_256 = _mm256_loadu_si256((__m256i*)&Bc[ki/2]);    //stores 8 bits twice (11,22,33....)
                __m512i b_expanded = _mm512_cvtepu8_epi16(b_packed_256);            //converted to 16 bits
                uint16_t tmp_bytes[32];
                _mm512_storeu_si512(tmp_bytes, b_expanded);
                uint16_t tmp_bf16[32];
                for (int t = 0; t < 32; ++t) {
                    int byte_idx = t >> 1;          // t / 2
                    uint8_t byte = (uint8_t)tmp_bytes[byte_idx];

                    uint8_t u4 = (t & 1)
                        ? (byte >> 4)               // odd → high nibble
                        : (byte & 0x0F);            // even → low nibble

                    tmp_bf16[t] = lut[u4];
                }
                __m512i b_bf16 = _mm512_loadu_si512(tmp_bf16);















                __m512 acc = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh)a, (__m512bh)b_bf16);


                float reduced = _mm512_reduce_add_ps(acc);


                uint32_t s_bits = ((uint32_t)Sc[ki/32]) << 16;
                float scale_f = *(float*)&s_bits;

                // printf("row %d col %d reduced %f  reduced*scale %f\n",row, col, reduced, reduced*scale_f);


                Cr[col] += reduced * scale_f;

            }
        }
    }
}
