#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include "ap_int.h"

#include "xcl2.hpp"
#include "cmdlineparser.h"
#include "llama.h"

using namespace sda;
using namespace sda::utils;

template <typename DType>
DType *GenerateRandomInput(unsigned int InputTokenLen = 16)
{
    DType *GeneratedInput = new DType[InputTokenLen];
    for (unsigned int IterRnd = 0; IterRnd < InputTokenLen; IterRnd++)
        GeneratedInput[IterRnd] = DType(rand());
    return GeneratedInput;
}

// template <typename DType>
// AttentionLayer::AttentionLayer();

// template <typename DType>
// AttentionLayer::AttentionLayer(PonyTensor<DType> WQIn, PonyTensor<DType> WKIn, PonyTensor<DType> WVIn)
// {
//     this->WQ = WQIn;
//     this->WK = WKIn;
//     this->WV = WVIn;
// };

// template <typename DType>
// FFNLayer::FFNLayer();

// template <typename DType>
// FFNLayer::FFNLayer(PonyTensor<DType> WIn)
// {
//     this->W = WIn;
// }

// template <typename DType>
// FFNLayer::forward(PonyTensor<DType> TokenInput, unsigned int TokenLen)
// {
// }

template <typename DType>
GemmRequest<DType>::GemmRequest(cl::Context &Ctx, cl::Program &Program, cl::CommandQueue &Queue)
{
    cl_int err;

    this->Q = Queue;

    OCL_CHECK(err, kernel = cl::Kernel(Program, "KrnlGemm", &err));

    OCL_CHECK(err, MatrixABuf = cl::Buffer(Ctx, CL_MEM_READ_ONLY, (GEMM_MAX_MATRIX_M * GEMM_MAX_MATRIX_N) * sizeof(DType), nullptr, &err));
    OCL_CHECK(err, MatrixBBuf = cl::Buffer(Ctx, CL_MEM_READ_ONLY, (GEMM_MAX_MATRIX_N * GEMM_MAX_MATRIX_K) * sizeof(DType), nullptr, &err));
    OCL_CHECK(err, MatrixResBuf = cl::Buffer(Ctx, CL_MEM_WRITE_ONLY, (GEMM_MAX_MATRIX_M * GEMM_MAX_MATRIX_K) * sizeof(DType), nullptr, &err));
    OCL_CHECK(err, err = kernel.setArg(3, MatrixABuf));
    OCL_CHECK(err, err = kernel.setArg(4, MatrixBBuf));
    OCL_CHECK(err, err = kernel.setArg(5, MatrixResBuf));

    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({MatrixABuf, MatrixBBuf, MatrixResBuf}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED));

    q.finish();
}

template <typename DType>
GemmRequest<DType>::run(PonyTensor<DType> &MatrixA, PonyTensor<DType> &MatrixB, PonyTensor<DType> &Res)
{
    assert(MatrixA.dim0 <= GEMM_MAX_MATRIX_M);
    assert(MatrixA.dim1 <= GEMM_MAX_MATRIX_N);
    assert(MatrixB.dim0 <= GEMM_MAX_MATRIX_N);
    assert(MatrixB.dim1 <= GEMM_MAX_MATRIX_K);
    assert(MatrixA.dim1 == MatrixB.dim0);

    cl_int err;
    cl::Event ReadEventMatrixA;
    cl::Event ReadEventMatrixB;
    cl::Event RunEvent;
    cl::Event ResEvent;

    this->finish();

    OCL_CHECK(err, err = kernel.setArg(0, MatrixA.dim0));
    OCL_CHECK(err, err = kernel.setArg(1, MatrixA.dim1));
    OCL_CHECK(err, err = kernel.setArg(2, MatrixB.dim1));

    OCL_CHECK(err, err = q.enqueueWriteBuffer(MatrixABuf, CL_FALSE, 0, (MatrixA.dim0 * MatrixA.dim1) * sizeof(DType), MatrixA.Data, nullptr, &ReadEventMatrixA));
    OCL_CHECK(err, err = q.enqueueWriteBuffer(MatrixBBuf, CL_FALSE, 0, (MatrixB.dim0 * MatrixB.dim1) * sizeof(DType), MatrixB.Data, nullptr, &ReadEventMatrixB));
    Events.push_back(ReadEventMatrixA);
    Events.push_back(ReadEventMatrixB);

    OCL_CHECK(err, err = q.enqueueTask(Kernel, &Events, &RunEvent));
    Events.push_back(RunEvent);

    OCL_CHECK(err, err = q.enqueueReadBuffer(MatrixResBuf, CL_FALSE, 0, (MatrixA.dim0 * MatrixB.dim1) * sizeof(DType), Res.Data, &Events, &ResEvent));
    Events.push_back(ResEvent);
    Res.dim0 = MatrixA.dim0;
    Res.dim1 = MatrixB.dim1;
}

template <typename DType>
GemmRequest<DType>::finish()
{
    if (events.size() > 0)
    {
        events.back().wait();
        events.clear();
        if (getenv("XCL_EMULATION_MODE") != NULL)
        {
            printf("Finished Gemm Request\n");
        }
    }
}

template <typename ReqType, typename DType>
KrnlDispatch<ReqType>::KrnlDispatch(cl::Context &Ctx, cl::Program &Program, cl::CommandQueue &Queue, unsigned int MaxReq)
{
    this->ReqNumMax = MaxReq;
    for (unsigned int IterReq = 0; IterReq < this->ReqNumMax; IterReq++)
        this->Reqs.push_back(ReqType(Ctx, Program, Queue));
}

template <typename ReqType, typename DType>
KrnlDispatch<ReqType>::request(PonyTensor<DType> &MatrixA, PonyTensor<DType> &MatrixB, PonyTensor<DType> &Res)
{
    this->Reqs[(this->Round++) % this->ReqNumMax].run(MatrixA, MatrixB, Res);
}

int main(int argc, char **argv)
{
    CmdLineParser parser;
    parser.addSwitch("--fpga", "-x", "FPGA binary (xclbin) file to use");

    parser.parse(argc, argv);
    string fpgaBinary = parser.value("fpga");

    if (fpgaBinary.size() == 0)
    {
        printf("ERROR: FPGA binary file (.xclbin) must be specified with the -x command line switch\n");
        return -1;
    }

    printf("FPGA binary       : %s\n", fpgaBinary.c_str());
    printf("Programming FPGA device\n");

    cl_int err;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    devices.resize(1);
    OCL_CHECK(err, cl::Context context(devices[0], NULL, NULL, NULL, &err));
    unsigned fileBufSize;
    char *fileBuf = xcl::read_binary_file(fpgaBinary.c_str(), fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue queue(context, devices[0], cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder, &err));

    KrnlDispatch<GemmRequest, int> GemmDispatch(context, program, queue, 1);
}
