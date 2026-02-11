#include <stdio.h>
#include <stdlib.h>
#include<stdint.h>
#include<time.h>
#include <math.h>
#include "src/avx512_kernel.h"
#include "src/avx512_matmul.h"
#include <omp.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))


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




void random_bf16_mat(uint16_t* A, int mn)
{
    for (int i = 0; i < mn; i++) {
        float x = ((rand() % 2000) - 1000) / 1000.0f;
        union { float f; uint32_t u; } t;
        t.f = x;
        A[i] = t.u >> 16;
    }
}

void random_int8_mat(uint8_t* B, int mn_1_2)            //input pointer to the matrix and the int 4 size divided by two so basically exact matrix size of B of int8 format
{
    for (int i = 0; i < mn_1_2; i++)
        B[i] = rand() & 0xFF;
}

int M[5] = {1, 4,8,16,32};                      //small shapes for the activation input 
int N[2] = {2048, 7168};                           //
int K[2] = {256*8, 896*8};                         //the K values aree for A matrix, for B we do K/2, and for S, we will do K/32

int m_r[5] = {1,2,4,8,16};
// int n_r[5];


int main()
{


    at::set_num_threads(1);
    at::set_num_interop_threads(1); 

    for(int i = 0; i < sizeof(M)/sizeof(int);i++)
    {
        for(int j = 0;j < sizeof(N)/sizeof(int);j++)
        {
            for(int k = 0;k < sizeof(K)/sizeof(int);k++)
            {

                for(int l = 0;l < sizeof(m_r)/sizeof(int);l++)
                {

                    int mr = min(m_r[l], M[i]);
                    int nr = min(m_r[l], N[j]);
                    
                    
                    uint16_t* A = (uint16_t*)malloc(M[i] * K[k] * sizeof(uint16_t));                            //A matrix, common for both torch and avx512
                    uint16_t* A_torch = (uint16_t*)malloc(M[i] * K[k] * sizeof(uint16_t));                            //A matrix, common for both torch and avx512
                    // uint16_t* A_torch = (uint16_t*)aligned_alloc(64, M[i] * K[k] * sizeof(uint16_t));
                    
                    
                    uint8_t* B = (uint8_t*)malloc((K[k] * N[j] / 2) * sizeof(uint8_t));                         //B matrix in int8 for avx512
                    uint16_t* B_torch = (uint16_t*)malloc((K[k] * N[j]) * sizeof(uint16_t));                         //B matrix in int8 for avx512
                    // uint16_t* B_torch = (uint16_t*)aligned_alloc(64, K[k] * N[j] * sizeof(uint16_t));
                    
                    
                    uint16_t* S = (uint16_t*)malloc((K[k] * N[j] / 32) * sizeof(uint16_t));                     //S matriix for avx512
                    
                    
                    float* C = (float*)malloc(M[i]*N[j]*sizeof(float));                                         //Avx512 output matrix
                    uint16_t* C_torch = (uint16_t*)malloc(M[i]*N[j]*sizeof(uint16_t));                                         //Avx512 output matrix

                    // uint16_t* C_torch = (uint16_t*)aligned_alloc(64, M[i] * N[j] * sizeof(uint16_t));
                    




                    random_bf16_mat(A, M[i] * K[k]);
                    random_bf16_mat(A_torch, M[i] * K[k]);

                    random_bf16_mat(S, (K[k] * N[j]) / 32);
                    random_bf16_mat(B_torch, K[k] * N[j]);
                    random_int8_mat(B, (K[k] * N[j]) / 2);




                    auto A_t = at::from_blob(
                        A_torch,
                        {M[i], K[k]},
                        at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU)
                    );

                    auto B_t = at::from_blob(
                        B_torch,
                        {K[k], N[j]},
                        at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU)
                    );

                    auto C_out = at::from_blob(
                        C_torch,
                        {M[i], N[j]},
                        at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU)
                    );


                    double avx512_gemm_start = omp_get_wtime();
                    matmul(A, B, S, C, M[i], N[j], K[k], 1, 1, lut);
                    double avx512_gemm_end = omp_get_wtime();



                    double torch_gemm_start = omp_get_wtime();
                    at::mm_out(C_out, A_t, B_t);
                    double torch_gemm_end = omp_get_wtime();


                    free(A);
                    free(A_torch);
                    free(B);
                    free(B_torch);
                    free(S);
                    free(C);
                    free(C_torch);

                    printf("M = %d  K = %d  N = %d  mr = %d  nr = %d AVX512 Time = %lf  Torch time = %lf  Speedup = %lf\n", M[i], K[k], N[j], mr, nr, avx512_gemm_end-avx512_gemm_start, torch_gemm_end - torch_gemm_start, (torch_gemm_end - torch_gemm_start)/(avx512_gemm_end-avx512_gemm_start));
                }
            }
        }
    }
}