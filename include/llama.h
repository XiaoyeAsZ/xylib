#ifndef __LLAMA__
#define __LLAMA__

#define LAYER_NUM 32
#define HEAD_NUM 32
#define EMBEDDING_DIM 4 * 1024
#define GEMM_MAX_MATRIX_M 4096
#define GEMM_MAX_MATRIX_N 4096
#define GEMM_MAX_MATRIX_K 4096

#include <vector>

template <typename DType>
struct PonyTensor
{
    DType *Data;
    unsigned int Dim0;
    unsigned int Dim1;
};

// template <typename DType>
// class AttentionLayer
// {
// private:
//     unsigned int CacheLen = 0;
//     unsigned int MaxCacheLen = 16;

//     PonyTensor<DType> WQ;
//     PonyTensor<DType> WK;
//     PonyTensor<DType> WV;
//     PonyTensor<DType> KCache;
//     PonyTensor<DType> VCache;

// public:
//     AttentionLayer();
//     AttentionLayer(PonyTensor<DType> WQIn, PonyTensor<DType> WKIn, PonyTensor<DType> WVIn);
//     ~AttentionLayer();

//     PonyTensor<DType> forward(PonyTensor<DType> TokenInput, unsigned int TokenLen);
// };

// template <typename DType>
// class FFNLayer
// {
// private:
//     PonyTensor<DType> W;

// public:
//     FFNLayer();
//     FFNLayer(PonyTensor<DType> WIn);
//     ~FFNLayer();

//     PonyTensor<DType> forward(PonyTensor<DType> TokenInput, unsigned int TokenLen);
// };

// template <typename DType>
// class GemvRequest
// {
// private:
//     /* data */
// public:
//     GemvRequest(/* args */);
//     ~GemvRequest();

//     void run();
// };

template <typename DType>
class GemmRequest
{
private:
    cl::Kernel Kernel;
    cl::CommandQueue Q;
    std::vector<cl::Event> Events;
    cl::Buffer MatrixABuf;
    cl::Buffer MatrixBBuf;
    cl::Buffer MatrixResBuf;

public:
    GemmRequest(cl::Context &Ctx, cl::Program &Program, cl::CommandQueue &Queue);
    ~GemmRequest();

    void run(PonyTensor<DType> &MatrixA, PonyTensor<DType> &MatrixB, PonyTensor<DType> &Res);
    void finish();
};

template <typename ReqType, typename DType>
class KrnlDispatch
{
private:
    std::vector<ReqType> Reqs;
    unsigned int ReqNumMax = 1;
    unsigned int Round = 0;

public:
    KrnlDispatch(cl::Context &Ctx, cl::Program &Program, cl::CommandQueue &Queue, unsigned int MaxReq);
    ~KrnlDispatch();

    void request(PonyTensor<DType> &MatrixA, PonyTensor<DType> &MatrixB, PonyTensor<DType> &Res);
};

#endif