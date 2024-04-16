#include <iostream>
#include <vector>
// #include "krnl_gemv.h"
// #include "krnl_gemm.h"
#include "krnl_addmat.h"
#include "ap_int.h"
#include <random>
#include <cmath>

#define MATRIX_SIZE 16

auto matrixA = new float[MATRIX_SIZE * MATRIX_SIZE];
auto matrixB = new float[MATRIX_SIZE * MATRIX_SIZE];
auto res = new float[MATRIX_SIZE * MATRIX_SIZE];
auto ref = new float[MATRIX_SIZE * MATRIX_SIZE];

int main()
{
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            matrixA[i * MATRIX_SIZE + j] = float(rand()) / RAND_MAX;
            matrixB[i * MATRIX_SIZE + j] = float(rand()) / RAND_MAX;
            // ref[i * MATRIX_SIZE + j] = matrixA[i * MATRIX_SIZE + j];
            // matrixB[i * MATRIX_SIZE + j] = float(rand()) / RAND_MAX;
            // matrixB[i * MATRIX_SIZE + j] = rand();
        }
    }

    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            ref[i * MATRIX_SIZE + j] = matrixA[i * MATRIX_SIZE + j] + matrixB[i * MATRIX_SIZE + j];
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

    // for (int i = 0; i < MATRIX_SIZE; i++)
    // {
    //     for (int j = 0; j < MATRIX_SIZE; j++)
    //     {
    //         std::cout << matrixA[i * MATRIX_SIZE + j] << " ";
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }

    KrnlAddMat(matrixA, 0, matrixB, 0, MATRIX_SIZE, MATRIX_SIZE, res, 0);

    bool flag = 1;
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            std::cout << ref[i * MATRIX_SIZE + j] << " : " << res[i * MATRIX_SIZE + j] << std::endl;
            if (abs(ref[i * MATRIX_SIZE + j] - res[i * MATRIX_SIZE + j]) > 1e-7)
            {
                flag = 0;
                // std::cout << ref[i * MATRIX_SIZE + j] << " : " << matrixB[j * MATRIX_SIZE + i] << std::endl;
            }
        }
        std::cout << std::endl;
    }

    std::cout << (flag ? "Test PASS !!! " : "ERROR !!! ") << std::endl;

    return 0;
}