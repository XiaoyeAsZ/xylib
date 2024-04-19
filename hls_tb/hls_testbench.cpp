#include <iostream>
#include <vector>
// #include "krnl_gemv.h"
// #include "krnl_gemm.h"
#include "krnl_gemv.h"
#include "ap_int.h"
#include <random>
#include <cmath>

#define MATRIX_SIZE 32

auto matrixA = new ap_int<8>[MATRIX_SIZE * MATRIX_SIZE];
auto vec = new ap_int<8>[MATRIX_SIZE];
auto res = new ap_int<8>[MATRIX_SIZE];
auto ref = new ap_int<8>[MATRIX_SIZE];

// auto matrixB = new ap_int<8>[MATRIX_SIZE * MATRIX_SIZE];
// auto res = new ap_int<8>[MATRIX_SIZE * MATRIX_SIZE];
// auto ref = new ap_int<8>[MATRIX_SIZE * MATRIX_SIZE];

int main()
{
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            int x = ((float(rand()) / RAND_MAX) - 0.5) * 255;
            // std::cout << x << std::endl;
            matrixA[i * MATRIX_SIZE + j] = x;

            // ref[i * MATRIX_SIZE + j] = matrixA[i * MATRIX_SIZE + j];
            // matrixB[i * MATRIX_SIZE + j] = float(rand()) / RAND_MAX;
            // matrixB[i * MATRIX_SIZE + j] = rand();
        }
        int y = ((float(rand()) / RAND_MAX) - 0.5) * 255;
        vec[i] = y;
    }

    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        ref[i] = 0;
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            std::cout<<matrixA[j * MATRIX_SIZE + i]<<" * "<<vec[j]<< " is : "<<ap_int<8>(matrixA[j * MATRIX_SIZE + i] * vec[j])<<std::endl;
            ref[i] += matrixA[i * MATRIX_SIZE + j] * vec[j];
        }
        std::cout << "correct: " << ref[i] << std::endl;
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

    ap_int<8> test[16][32 * 32 / 16];

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            for (int k = 0; k < 32; k++)
                test[j][k + i * 32] = matrixA[(j + i * 16) * MATRIX_SIZE + k];
        }
    }

    for (int i = 0; i < 32; i++)
    {
        for (int j = 0; j < 32; j++)
            std::cout << matrixA[i * MATRIX_SIZE + j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;
    for (int i = 0; i < 32; i++)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < 32; i++)
    {
        std::cout << ap_int<8>(((PACK_INT8_32 *)(vec))[0](i * 8 + 7, i * 8)) << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // std::cout << std::endl;

    // for (int i = 0; i < 16; i++)
    // {
    //     for (int j = 0; j < 256; j++)
    //         std::cout << test[i][j] << " ";
    //     std::cout << std::endl;
    // }

    KrnlGemv((PACK_INT8_32 *)(&test[0]), (PACK_INT8_32 *)(&test[1]), (PACK_INT8_32 *)(&test[2]), (PACK_INT8_32 *)(&test[3]),
             (PACK_INT8_32 *)(&test[4]), (PACK_INT8_32 *)(&test[5]), (PACK_INT8_32 *)(&test[6]), (PACK_INT8_32 *)(&test[7]),
             (PACK_INT8_32 *)(&test[8]), (PACK_INT8_32 *)(&test[9]), (PACK_INT8_32 *)(&test[10]), (PACK_INT8_32 *)(&test[11]),
             (PACK_INT8_32 *)(&test[12]), (PACK_INT8_32 *)(&test[13]), (PACK_INT8_32 *)(&test[14]), (PACK_INT8_32 *)(&test[15]),
             (PACK_INT8_32 *)(vec), (PACK_INT8_16 *)(res), MATRIX_SIZE, MATRIX_SIZE);

    bool flag = 1;
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        // for (int j = 0; j < MATRIX_SIZE; j++)
        // {
        std::cout << ref[i] << " : " << res[i] << std::endl;
        if (ref[i] != res[i])
        {
            flag = 0;
            // std::cout << ref[i * MATRIX_SIZE + j] << " : " << matrixB[j * MATRIX_SIZE + i] << std::endl;
        }
        // }
        std::cout << std::endl;
    }

    std::cout << (flag ? "Test PASS !!! " : "ERROR !!! ") << std::endl;

    return 0;
}