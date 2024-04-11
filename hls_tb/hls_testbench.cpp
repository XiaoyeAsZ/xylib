#include <iostream>
#include <vector>
// #include "krnl_gemv.h"
// #include "krnl_gemm.h"
#include "krnl_transpose.h"
#include "ap_int.h"
#include <random>

#define MATRIX_SIZE 16

auto matrixA = new ap_int<32>[MATRIX_SIZE * MATRIX_SIZE];
auto matrixB = new ap_int<32>[MATRIX_SIZE * MATRIX_SIZE];
auto res = new ap_int<32>[MATRIX_SIZE * MATRIX_SIZE];
auto ref = new ap_int<32>[MATRIX_SIZE * MATRIX_SIZE];

int main()
{
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            matrixA[i * MATRIX_SIZE + j] = i - j;
            // matrixB[i * MATRIX_SIZE + j] = rand();
        }
    }

    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            ap_int<32> tmp = 0;
            for (int k = 0; k < MATRIX_SIZE; k++)
                tmp += matrixA[i * MATRIX_SIZE + k] * matrixB[k * MATRIX_SIZE + j];
            ref[i * MATRIX_SIZE + j] = tmp;
        }
    }

    // KrnlGemm(MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, matrixA, matrixB, res);

    // int ecnt = 0;
    // bool flag = 1;
    // for (int i = 0; i < MATRIX_SIZE; i++)
    // {
    //     for (int j = 0; j < MATRIX_SIZE; j++)
    //     {
    //         std::cout << ref[i * MATRIX_SIZE + j] << " ";
    //         if (ref[i * MATRIX_SIZE + j] != res[i * MATRIX_SIZE + j])
    //         {
    //             flag = 0;
    //             ecnt++;
    //         }
    //     }
    //     std::cout << std::endl;
    // }

    // for (int i = 0; i < MATRIX_SIZE; i++)
    // {
    //     for (int j = 0; j < MATRIX_SIZE; j++)
    //     {
    //         std::cout << res[i * MATRIX_SIZE + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    KrnlTranspose(MATRIX_SIZE, MATRIX_SIZE, matrixA, matrixB);

    bool flag = 1;
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            std::cout << matrixB[i * MATRIX_SIZE + j] << " ";

            if (matrixA[i * MATRIX_SIZE + j] != matrixB[j * MATRIX_SIZE + i])
            {
                flag = 0;
                std::cout << matrixA[i * MATRIX_SIZE + j] << " : " << matrixB[j * MATRIX_SIZE + i] << std::endl;
            }
        }
        std::cout << std::endl;
    }

    std::cout << (flag ? "Test PASS !!! " : "ERROR !!! ") << std::endl;

    return 0;
}