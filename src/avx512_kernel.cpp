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

    int N              // for testing purposes rn
)
{

    //step 1- load the lookup table in a register

    // __m256i lut256 = _mm256_loadu_si256((__m256i*)lut);
    // __m512i lut_r  = _mm512_zextsi256_si512(lut256);                //zero extend a 256 bit register (16*16 values + 0s)

    __m512i lut_r = _mm512_mask_loadu_epi16(_mm512_undefined_si512(), 0xFFFF, lut);    //1111 1111 1111 1111, basically take first 16 values form lut, and dont care about the rest

    __m128i mask00001111 = _mm_set1_epi8(0x0F);


    for(int row = 0;row < r;row+=4)
    {
        const uint16_t* Ar = A + row*k;
        const uint16_t* Ar_1 = A + (row+1)*k;
        const uint16_t* Ar_2 = A + (row+2)*k;
        const uint16_t* Ar_3 = A + (row+3)*k;

        float* Cr = C + (row)*N;
        float* Cr_1 = C + (row+1)*N;
        float* Cr_2 = C + (row+2)*N;
        float* Cr_3 = C + (row+3)*N;



        for(int col = 0; col < c;col += 2)
        {
            const uint8_t* Bc = B + (col * k)/2;
            const uint8_t* Bc_1 = B + ((col+1) * k)/2;

            const uint16_t* Sc = S + (col * k)/32;
            const uint16_t* Sc_1 = S + ((col+1) * k)/32;


            __m512 acc = _mm512_setzero_ps();
            __m512 acc_1 = _mm512_setzero_ps();
            __m512 acc_2 = _mm512_setzero_ps();
            __m512 acc_3 = _mm512_setzero_ps();

            __m512 acc_0_1 = _mm512_setzero_ps();
            __m512 acc_1_1 = _mm512_setzero_ps();
            __m512 acc_2_1 = _mm512_setzero_ps();
            __m512 acc_3_1 = _mm512_setzero_ps();




            for(int ki = 0;ki < k;ki += 32)
            {

                if (ki + 32 < k) {
                    _mm_prefetch((const char*)&Ar[ki + 32], _MM_HINT_T0);
                    _mm_prefetch((const char*)&Ar_1[ki + 32], _MM_HINT_T0);
                    _mm_prefetch((const char*)&Ar_2[ki + 32], _MM_HINT_T0);
                    _mm_prefetch((const char*)&Ar_3[ki + 32], _MM_HINT_T0);
                    _mm_prefetch((const char*)&Bc[ki/2 + 16], _MM_HINT_T0);
                    _mm_prefetch((const char*)&Bc_1[ki/2 + 16], _MM_HINT_T0);
                }
                __m128i b_128 = _mm_loadu_epi8((void*)&Bc[ki/2]);        //32 int4s as 16 int8s stored in b_128, b0, b1, b2 , .. b16

                __m128i b_hi_128 = _mm_srli_epi16(b_128, 4);                 //right shifts to get the first 4 bits of bs
                b_hi_128 = _mm_and_si128(b_hi_128, mask00001111);         //shift operation is for 16bit, so two bytes are hifted right at once, then we take and with 00001111 so it is fixed

                __m128i b_lo_128 = _mm_and_si128(b_128, mask00001111);       //bitwise and with 00001111


                /* 
                    b_lo_128 is b0, b1, b2, ... b16's lower 4 bits stores as 0000xxxx format 
                    b_hi_128 is b0, b1, b2, ....b16's higher 4 bits stored as 0000xxxx format
                */


                __m128i lo = _mm_unpacklo_epi8(b_lo_128, b_hi_128);         // b_lo(0) b_hi(0) b_lo(1) b_hi(1) ... b_lo(7) b_hi(7)
                b_hi_128 = _mm_unpackhi_epi8(b_lo_128, b_hi_128);           // b_lo(8) b_hi(8) b_lo(9) b_hi(9) ... b_lo(15) b_hi(15)

                __m256i b_256 = _mm256_set_m128i(b_hi_128, lo);             // contacatonate the above two

                //in the last three instructions, actullay, we didnt make one new hi register, and used the previous register again and thus saved up one more register

                __m512i b = _mm512_cvtepu8_epi16(b_256);                         //finally convert to 16 bit and store in 512 bit register
                b = _mm512_permutexvar_epi16(b, lut_r);                 //the nibbles are treates as indiecs and then we do a lookup and get bf16 bits




                __m128i b_128_1 = _mm_loadu_epi8((void*)&Bc_1[ki/2]);        //32 int4s as 16 int8s stored in b_128, b0, b1, b2 , .. b16

                __m128i b_hi_128_1 = _mm_srli_epi16(b_128_1, 4);                 //right shifts to get the first 4 bits of bs
                b_hi_128_1 = _mm_and_si128(b_hi_128_1, mask00001111);         //shift operation is for 16bit, so two bytes are hifted right at once, then we take and with 00001111 so it is fixed

                __m128i b_lo_128_1 = _mm_and_si128(b_128_1, mask00001111);       //bitwise and with 00001111


                /* 
                    b_lo_128 is b0, b1, b2, ... b16's lower 4 bits stores as 0000xxxx format 
                    b_hi_128 is b0, b1, b2, ....b16's higher 4 bits stored as 0000xxxx format
                */


                __m128i lo_1 = _mm_unpacklo_epi8(b_lo_128_1, b_hi_128_1);         // b_lo(0) b_hi(0) b_lo(1) b_hi(1) ... b_lo(7) b_hi(7)
                b_hi_128_1 = _mm_unpackhi_epi8(b_lo_128_1, b_hi_128_1);           // b_lo(8) b_hi(8) b_lo(9) b_hi(9) ... b_lo(15) b_hi(15)

                __m256i b_256_1 = _mm256_set_m128i(b_hi_128_1, lo_1);             // contacatonate the above two

                //in the last three instructions, actullay, we didnt make one new hi register, and used the previous register again and thus saved up one more register

                __m512i b_1 = _mm512_cvtepu8_epi16(b_256_1);                         //finally convert to 16 bit and store in 512 bit register
                b_1 = _mm512_permutexvar_epi16(b_1, lut_r);                 //the nibbles are treates as indiecs and then we do a lookup and get bf16 bits




                
                uint32_t s_bits = ((uint32_t)Sc[ki/32]) << 16;                  //float is bf16 with 16 more decimals so just shift left a bf16 to get float
                float scale_f = *(float*)&s_bits;

                __m512 scale_broadcast = _mm512_set1_ps(scale_f);


                uint32_t s_bits_1 = ((uint32_t)Sc_1[ki/32]) << 16;                  //float is bf16 with 16 more decimals so just shift left a bf16 to get float
                float scale_f_1 = *(float*)&s_bits_1;

                __m512 scale_broadcast_1 = _mm512_set1_ps(scale_f_1);


        





                __m512i a = _mm512_load_si512((void*)&Ar[ki]);
                __m512 acc_tmp = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh)a, (__m512bh)b);    
                acc = _mm512_fmadd_ps(acc_tmp, scale_broadcast, acc);
                
                __m512 acc_tmp_0_1 = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh)a, (__m512bh)b_1);    
                acc_0_1 = _mm512_fmadd_ps(acc_tmp_0_1, scale_broadcast_1, acc_0_1);


                __m512i a_1 = _mm512_load_si512((void*)&Ar_1[ki]);
                __m512 acc_tmp_1 = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh)a_1, (__m512bh)b);   
                acc_1 = _mm512_fmadd_ps(acc_tmp_1, scale_broadcast, acc_1);
             
                __m512 acc_tmp_1_1 = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh)a_1, (__m512bh)b_1);   
                acc_1_1 = _mm512_fmadd_ps(acc_tmp_1_1, scale_broadcast_1, acc_1_1);



                __m512i a_2 = _mm512_load_si512((void*)&Ar_2[ki]);
                __m512 acc_tmp_2 = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh)a_2, (__m512bh)b);               
                acc_2 = _mm512_fmadd_ps(acc_tmp_2, scale_broadcast, acc_2);

               __m512 acc_tmp_2_1 = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh)a_2, (__m512bh)b_1);               
                acc_2_1 = _mm512_fmadd_ps(acc_tmp_2_1, scale_broadcast_1, acc_2_1);



                __m512i a_3 = _mm512_load_si512((void*)&Ar_3[ki]);
                __m512 acc_tmp_3 = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh)a_3, (__m512bh)b);               
                acc_3 = _mm512_fmadd_ps(acc_tmp_3, scale_broadcast, acc_3);
 
                __m512 acc_tmp_3_1 = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh)a_3, (__m512bh)b_1);               
                acc_3_1 = _mm512_fmadd_ps(acc_tmp_3_1, scale_broadcast_1, acc_3_1);








                //here also, when b is declared, it actually stores the indices for the lookup table, then we use the same register to store the actual bf16s


            }


            Cr[col] = _mm512_reduce_add_ps(acc);
            Cr_1[col] = _mm512_reduce_add_ps(acc_1);
            Cr_2[col] = _mm512_reduce_add_ps(acc_2);
            Cr_3[col] = _mm512_reduce_add_ps(acc_3);

            Cr[col+1] = _mm512_reduce_add_ps(acc_0_1);
            Cr_1[col+1] = _mm512_reduce_add_ps(acc_1_1);
            Cr_2[col+1] = _mm512_reduce_add_ps(acc_2_1);
            Cr_3[col+1] = _mm512_reduce_add_ps(acc_3_1);


        }
    }
}
