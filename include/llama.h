#ifndef __LLAMA__
#define __LLAMA__

#define LAYER_NUM 32
#define HEAD_NUM 32
#define EMBEDDING_DIM 64
#define TEST_TOKEN_LEN 16
#define GEMM_MAX_MATRIX_M 4096
#define GEMM_MAX_MATRIX_N 4096
#define GEMM_MAX_MATRIX_K 4096

#include <vector>

struct TensorOnFPGA
{
    cl::Buffer Data;
    unsigned int Offset;
    unsigned int Dim0;
    unsigned int Dim1;

    TensorOnFPGA()
    {
        Offset = 0;
        Dim0 = 0;
        dim1 = 0;
    }

    void ReleaseMem()
    {
        clReleaseMemObject(Data);
    }

    TensorOnFPGA SubTensorRow(unsigned int Start, unsigned int Len)
    {
        return TensorOnFPGA{Data, Start, Len, Dim1};
    }
};

template <typename DType>
struct TensorOnHost
{
    DType *Data;
    unsigned int Dim0;
    unsigned int Dim1;

    TensorOnHost()
    {
        Dim0 = 0;
        dim1 = 0;
    }
};

class OCLWrap
{
private:
    cl::Context Ctx;
    cl::CommandQueue Queue;

    cl::Kernel KrnlAdd;

public:
    template <typename DType>
    cl::Buffer AllocateReadBuffer(unsigned int Size);

    template <typename DType>
    cl::Buffer AllocateWriteBuffer(unsigned int Size);

    template <typename DType>
    cl::Buffer AllocateReadWriteBuffer(unsigned int Size);

    template <typename DType>
    TensorOnFPGA Mul(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1, std::vector<cl::Event> &Events);

    template <typename DType>
    TensorOnFPGA Add(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1, std::vector<cl::Event> &Events);

    template <typename DType>
    TensorOnFPGA Dot(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1, std::vector<cl::Event> &Events);

    template <typename DType>
    TensorOnFPGA Silu(TensorOnFPGA &Tensor0, std::vector<cl::Event> &Events);

    TensorOnFPGA Freq;
    template <typename DType>
    TensorOnHost REmb(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1, std::vector<cl::Event> &Events);

    template <typename DType>
    void Move(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1, std::vector<cl::Event> &Events);

    template <typename DType>
    TensorOnFPGA Trans(TensorOnFPGA &Tensor0, std::vector<cl::Event> &Events);

    template <typename DType>
    TensorOnFPGA Softmax(TensorOnFPGA &Tensor0, std::vector<cl::Event> &Events);

    template <typename DType>
    TensorOnFPGA RMSNorm(TensorOnFPGA &Tensor0, std::vector<cl::Event> &Events);

}

template <typename DType>
class AttentionLayer
{
private:
    OCLWrap &OCL;

    std::vector<TensorOnHost<DType>> WQHost;
    std::vector<TensorOnHost<DType>> WKHost;
    std::vector<TensorOnHost<DType>> WVHost;
    TensorOnHost<DType> WOHost;
    std::vector<TensorOnFPGA> WQFPGA;
    std::vector<TensorOnFPGA> WKFPGA;
    std::vector<TensorOnFPGA> WVFPGA;
    TensorOnFPGA WOFPGA;
    std::vector<TensorOnFPGA> KCache;
    std::vector<TensorOnFPGA> VCache;

    unsigned int MaxCacheLen = 16;
    unsigned int CurLen = 0;

public:
    AttentionLayer();
    ~AttentionLayer();

    void load(std::ifstream &F);
    TensorOnFPGA operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events);
};

template <typename DType>
class FeedForwardLayer
{
private:
    OCLWrap &OCL;

    TensorOnHost<DType> W0Host;
    TensorOnHost<DType> W1Host;
    TensorOnHost<DType> W2Host;
    TensorOnFPGA W0FPGA;
    TensorOnFPGA W1FPGA;
    TensorOnFPGA W2FPGA;

public:
    FeedForwardLayer();
    ~FeedForwardLayer();

    void load(std::ifstream &F);
    TensorOnFPGA operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events);
};

template <typename DType>
class RMSNormLayer
{
private:
    TensorOnFPGA Ep;
    TensorOnFPGA Weight;

public:
    RMSNormLayer();
    ~RMSNormLayer();

    TensorOnFPGA operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events);
};

template <typename DType>
class TransformerBlock
{
private:
    OCLWrap &OCL;

    AttentionLayer<DType> Attn;
    FeedForwardLayer<DType> FFN;
    RMSNormLayer<DType> NormAttn;
    RMSNormLayer<DType> NormFFN;

public:
    TransformerBlock();
    ~TransformerBlock();

    void load(std::ifstream &F);
    TensorOnFPGA operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events);
};

template <typename DType>
class Transformer
{
private:
    OCLWrap &OCL;

    std::vector<TransformerBlock<DType>> Blocks;

public:
    Transformer();
    ~Transformer();

    void load(std::ifstream &F);
    TensorOnFPGA operator()(TensorOnFPGA Input, std::vector<cl::Event> &Events);
};
