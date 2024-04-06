#include <iostream>
#include <vector>
// #include "krnl_gemv.h"
#include "krnl_gemm.h"
#include "ap_int.h"

// auto matrix = new ap_int<32>[128 * 128];
// auto vector = new ap_int<32>[128];
// auto res = new ap_int<32>[128];

// auto ref = new ap_int<32>[128];

// int main()
// {
//     for (int i = 0; i < 128; i++)
//     {
//         vector[i] = i;
//         for (int j = 0; j < 128; j++)
//             matrix[i * 128 + j] = i + j;
//     }

//     for (int i = 0; i < 128; i++)
//     {
//         ap_int<32> tmp = 0;
//         for (int j = 0; j < 128; j++)
//             tmp += matrix[i * 128 + j] * vector[j];
//         ref[i] = tmp;
//     }

//     KrnlGemv(128, 128, matrix, vector, res);

//     bool flag = 1;
//     for (int i = 0; i < 128; i++)
//     {
//         if (res[i] != ref[i])
//             flag = 0;
//     }

//     std::cout << (flag ? "Test PASS !!!" : "ERROR !!!") << std::endl;

//     return 0;
// }

auto matrixA = new ap_int<32>[128 * 128];
auto matrixB = new ap_int<32>[128 * 128];
auto res = new ap_int<32>[128 * 128];
auto ref = new ap_int<32>[128 * 128];

int main()
{
    for (int i = 0; i < 128; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            matrixA[i * 128 + j] = i + j;
            matrixB[i * 128 + j] = i - j;
        }
    }

    for (int i = 0; i < 128; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            ap_int<32> tmp = 0;
            for (int k = 0; k < 128; k++)
                tmp += matrixA[i * 128 + k] * matrixB[k * 128 + j];
            ref[i * 128 + j] = tmp;
        }
    }

    KrnlGemm(128, 128, 128, matrixA, matrixB, res);

    bool flag = 1;
    for (int i = 0; i < 128; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            if (res[i * 128 + j] != ref[i * 128 + j])
                flag = 0;
        }
    }

    std::cout << (flag ? "Test PASS !!!" : "ERROR !!!") << std::endl;

    return 0;
}