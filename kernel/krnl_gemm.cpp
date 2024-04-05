/**
 *
 *
 *
 */

#include "hls_stream.h"
#include "types.hpp"
#include "krnl_gemm.h"
#include "u280.h"
#include <iostream>


extern "C"
{
    extern "C"
    {
        void KrnlGemm(
            const unsigned int DimM,
            const unsigned int DimN,
            const unsigned int DimK,
            DATA_TYPE *MatrixAInMem,
            DATA_TYPE *MatrixBInMem,
            RES_TYPE *MatrixRes);
    }
}
