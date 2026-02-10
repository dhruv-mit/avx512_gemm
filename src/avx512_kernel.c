#include "avx512_kernel.h"


void kernel_bf16_int4_bf16(
    const uint16_t* A,             //pointer to start of sub matrix of A
    const uint8_t * B,              //pointer to start of sub matrix of B(2 elements per 8 bit)
    const uint16_t* S,               //pointer to start of submatrix of sclae matrix
    float* C,                        //pointer to start of sub matrix of C
    int r,                           //number of rows in submatrix of A
    int c,                           //number of cols in submatrix of B
    int k,                           //common dim
    const uint16_t* lut,              //lookup table(maps 16 int4s to bf16s)

    int col_size_of_c              // for testing purposes rn
)
{

    //step 1- load the lookup table in a register

    // __m512i lut_r = _mm512_loadu_epi16((void*)lut);             //probably can store in a 256i register

    __m256i lut256 = _mm256_loadu_si256((__m256i*)lut);
    __m512i lut_r  = _mm512_zextsi256_si512(lut256);                //zero extend a 256 bit register

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


                __m128i b_128 = _mm_loadu_epi8((void*)&Bc[ki/2]);        //16 int8s or 32 int4s stored in b_128

                __m256i tmp256 = _mm256_cvtepu8_epi16(b_128);       //store uint8 as uint16

                __m512i wide = _mm512_zextsi256_si512(tmp256);      //kind of like typecast into 512 bit

                __m512i idx = _mm512_set_epi16(
                    15,15,14,14,13,13,12,12,
                    11,11,10,10,9,9,8,8,
                    7,7,6,6,5,5,4,4,
                    3,3,2,2,1,1,0,0);

                __m512i b_512 = _mm512_permutexvar_epi16(idx, wide);


                __m512i shifted = _mm512_srli_epi16(b_512, 4);    //shift right by 4

                __m512i mask = _mm512_set1_epi16(0x000F);
                __m512i masked = _mm512_and_si512(b_512, mask);

                __mmask32 odd = 0xAAAAAAAA;                     //1010101010 mask


                __m512i b_idx = _mm512_mask_mov_epi16(masked, odd, shifted);

                // uint16_t print_buf[32];

    


                __m512i b = _mm512_permutexvar_epi16(b_idx, lut_r);

                __m512 acc = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh)a, (__m512bh)b);               //currently this wont work because b is not in bf16 format


                //  _mm512_store_si512(print_buf, b);

                // for(int i = 0;i < 32;i++)
                // {
                //     printf("%u ", lut_r[i]);
                // }


                float reduced = _mm512_reduce_add_ps(acc);


                uint32_t s_bits = ((uint32_t)Sc[ki/32]) << 16;
                float scale_f = *(float*)&s_bits;

                // printf("row %d col %d reduced %f  reduced*scale %f\n",row, col, reduced, reduced*scale_f);


                Cr[col] += reduced * scale_f;

            }
        }
    }
}
