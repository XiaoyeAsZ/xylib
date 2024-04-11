#include "hls_stream.h"
#include "types.hpp"
#include "krnl_silu.h"
#include "u280.h"
#include <iostream>

extern "C"
{
    void KrnlSilu(DATA_TYPE *Matrix,
                  const unsigned int Offeset,
                  const unsigned int DimM,
                  const unsigned int DimN,
                  DATA_TYPE *MatrixTrans)
    {
        
    }
}