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
  DType *GeneratedInput = new DType[InputTokenLen * EMBEDDING_DIM];
  for (unsigned int IterRnd = 0; IterRnd < InputTokenLen * EMBEDDING_DIM; IterRnd++)
    GeneratedInput[IterRnd] = DType(rand());
  return GeneratedInput;
}

template <typename DType>
TensorOnHost<DType> operator*(TensorOnHost<DType> &MA, TensorOnHost<DType> &MB)
{
  assert(MA.Dim1 == MB.Dim0);
  TensorOnHost<DType> Res;
  for (int i = 0; i < MA.Dim0; i++)
  {
    for (int j = 0; i < MB.Dim1; i++)
    {
      DType tmp = 0;
      for (int k = 0; k < MA.Dim1; k++)
        tmp += MA.Data[i * MA.Dim1 + k] * MB.Data[k * MB.Dim1 + j];
      Res.Data[i * MB.Dim1 + j] = tmp;
    }
  }
  return Res;
}

template <typename DType>
bool operator==(TensorOnHost<DType> &MA, TensorOnHost<DType> &MB)
{
  if (MA.Dim0 != MB.Dim0 || MA.Dim1 != MB.Dim1)
    return 0;
  for (int i = 0; i < MA.Dim0 * MA.Dim1; i++)
  {
    if (MA.Data[i] != MB.Data[i])
      return 0;
  }
  return 1;
}

template <typename DType>
AttentionHead::AttentionHead(){};

template <typename DType>
AttentionHead::AttentionHead(TensorOnHost<DType> &WQIn, TensorOnHost<DType> &WKIn, TensorOnHost<DType> &WVIn)
{
  this->WQHost = WQIn;
  this->WKHost = WKIn;
  this->WVHost = WVIn;
};

template <typename DType>
AttentionHead::~AttentionHead(){};

template <typename DType>
void AttentionHead::migrate(cl::Context Ctx, cl::CommandQueue Queue)
{
  cl_int err;
  OCL_CHECK(err, this->WQFPGA = cl::Buffer(Ctx, CL_MEM_READ_ONLY, (EMBEDDING_DIM * EMBEDDING_DIM) * sizeof(DType), nullptr, &err));
  OCL_CHECK(err, this->WKFPGA = cl::Buffer(Ctx, CL_MEM_READ_ONLY, (EMBEDDING_DIM * EMBEDDING_DIM) * sizeof(DType), nullptr, &err));
  OCL_CHECK(err, this->WVFPGA = cl::Buffer(Ctx, CL_MEM_READ_ONLY, (EMBEDDING_DIM * EMBEDDING_DIM) * sizeof(DType), nullptr, &err));
  OCL_CHECK(err, err = Queue.enqueueWriteBuffer(this->WQFPGA.Data, CL_FALSE, 0, (this->WQFPGA.Dim0 * this->WQFPGA.Dim1) * sizeof(DType), this->WQHost.Data, nullptr));
  OCL_CHECK(err, err = Queue.enqueueWriteBuffer(this->WKFPGA.Data, CL_FALSE, 0, (this->WKFPGA.Dim0 * this->WKFPGA.Dim1) * sizeof(DType), this->WKHost.Data, nullptr));
  OCL_CHECK(err, err = Queue.enqueueWriteBuffer(this->WVFPGA.Data, CL_FALSE, 0, (this->WVFPGA.Dim0 * this->WVFPGA.Dim1) * sizeof(DType), this->WVHost.Data, nullptr));
  Queue.finish();
};

template <typename DType>
TensorOnFPGA AttentionHead::operator()(TensorOnFPGA &Input, KrnlDispatch &Dispatch){

};

template <typename DType>
GemmRequest<DType>::GemmRequest(cl::Context &Ctx, cl::Program &Program, cl::CommandQueue &Queue)
{
  cl_int err;

  this->Q = Queue;

  OCL_CHECK(err, this->Kernel = cl::Kernel(Program, "KrnlGemm", &err));

  // OCL_CHECK(err, MatrixABuf = cl::Buffer(Ctx, CL_MEM_READ_ONLY, (GEMM_MAX_MATRIX_M * GEMM_MAX_MATRIX_N) * sizeof(DType), nullptr, &err));
  // OCL_CHECK(err, MatrixBBuf = cl::Buffer(Ctx, CL_MEM_READ_ONLY, (GEMM_MAX_MATRIX_N * GEMM_MAX_MATRIX_K) * sizeof(DType), nullptr, &err));
  // OCL_CHECK(err, MatrixResBuf = cl::Buffer(Ctx, CL_MEM_WRITE_ONLY, (GEMM_MAX_MATRIX_M * GEMM_MAX_MATRIX_K) * sizeof(DType), nullptr, &err));
  // OCL_CHECK(err, err = kernel.setArg(3, MatrixABuf));
  // OCL_CHECK(err, err = kernel.setArg(4, MatrixBBuf));
  // OCL_CHECK(err, err = kernel.setArg(5, MatrixResBuf));

  // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({MatrixABuf, MatrixBBuf, MatrixResBuf}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED));

  // this->Q.finish();
}

template <typename DType>
void GemmRequest<DType>::run(TensorOnFPGA &MatrixA, TensorOnFPGA &MatrixB, TensorOnFPGA &Res)
{
  assert(MatrixA.Dim0 <= GEMM_MAX_MATRIX_M);
  assert(MatrixA.Dim1 <= GEMM_MAX_MATRIX_N);
  assert(MatrixB.Dim0 <= GEMM_MAX_MATRIX_N);
  assert(MatrixB.Dim1 <= GEMM_MAX_MATRIX_K);
  assert(MatrixA.Dim1 == MatrixB.Dim0);

  cl_int err;
  cl::Event RunEvent;

  this->finish();

  OCL_CHECK(err, err = this->Kernel.setArg(0, MatrixA.Dim0));
  OCL_CHECK(err, err = this->Kernel.setArg(1, MatrixA.Dim1));
  OCL_CHECK(err, err = this->Kernel.setArg(2, MatrixB.Dim1));
  OCL_CHECK(err, err = this->Kernel.setArg(3, MatrixA.Data));
  OCL_CHECK(err, err = this->Kernel.setArg(4, MatrixB.Data));
  OCL_CHECK(err, err = this->Kernel.setArg(5, Res.Data));

  // OCL_CHECK(err, err = q.enqueueWriteBuffer(MatrixABuf, CL_FALSE, 0, (MatrixA.Dim0 * MatrixA.Dim1) * sizeof(DType), MatrixA.Data, nullptr, &ReadEventMatrixA));
  // OCL_CHECK(err, err = q.enqueueWriteBuffer(MatrixBBuf, CL_FALSE, 0, (MatrixB.Dim0 * MatrixB.Dim1) * sizeof(DType), MatrixB.Data, nullptr, &ReadEventMatrixB));
  // Events.push_back(ReadEventMatrixA);
  // Events.push_back(ReadEventMatrixB);

  OCL_CHECK(err, err = this->Q.enqueueTask(this->Kernel, &this->Events, &RunEvent));
  this->Events.push_back(RunEvent);

  // OCL_CHECK(err, err = q.enqueueReadBuffer(MatrixResBuf, CL_FALSE, 0, (MatrixA.Dim0 * MatrixB.Dim1) * sizeof(DType), Res.Data, &Events, &ResEvent));
  // Events.push_back(ResEvent);
  Res.Dim0 = MatrixA.Dim0;
  Res.Dim1 = MatrixB.Dim1;
}

template <typename DType>
void GemmRequest<DType>::finish()
{

  if (getenv("XCL_EMULATION_MODE") != NULL)
  {
    printf("Finished Gemm Request\n");
  }
}

template <typename DType>
GemmDispatch<DType>::GemmDispatch(cl::Context &Ctx, cl::Program &Program, cl::CommandQueue &Queue, unsigned int MaxReq)
{
  this->ReqNumMax = MaxReq;
  for (unsigned int IterReq = 0; IterReq < this->ReqNumMax; IterReq++)
    this->Reqs.push_back(GemmRequest<DType>(Ctx, Program, Queue));
}

template <typename DType>
void GemmDispatch<DType>::request(TensorOnFPGA &MatrixA, TensorOnFPGA &MatrixB, TensorOnFPGA &Res)
{
  this->Reqs[(this->Round++) % this->ReqNumMax].run(MatrixA, MatrixB, Res);
}

/**
 * Class OCLWrap
 *
 */

template <typename DType>
cl::Buffer OCLWrap::AllocateReadBuffer(unsigned int Size)
{
  cl_int err;
  cl::Buffer Buf;
  OCL_CHECK(err, Buf = cl::Buffer(this->Ctx, CL_MEM_READ_ONLY, (Size) * sizeof(DType), nullptr, &err));
  return Buf;
};

template <typename DType>
cl::Buffer OCLWrap::AllocateWriteBuffer(unsigned int Size)
{
  cl_int err;
  cl::Buffer Buf;
  OCL_CHECK(err, Buf = cl::Buffer(this->Ctx, CL_MEM_WRITE_ONLY, (Size) * sizeof(DType), nullptr, &err));
  return Buf;
};

template <typename DType>
cl::Buffer OCLWrap::AllocateReadWriteBuffer(unsigned int Size)
{
  cl_int err;
  cl::Buffer Buf;
  OCL_CHECK(err, Buf = cl::Buffer(this->Ctx, CL_MEM_READ_WRITE, (Size) * sizeof(DType), nullptr, &err));
  OCL_CHECK(err, err = this->Queue.enqueueMigrateMemObjects({Buf}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED));
  this->Queue.finish();
  return Buf;
};

template <typename DType>
TensorOnFPGA OCLWrap::Add(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1, std::vector<cl::Event> &Events)
{
  assert(Tensor0.Dim0 == Tensor1.Dim0 && Tensor0.Dim1 == Tensor1.Dim1);

  cl_int err;
  cl::Event RunEvent;
  TensorOnFPGA Buf = {this->AllocateReadWriteBuffer<DType>(Tensor0.Dim0 * Tensor0.Dim1), Tensor0.Dim0, Tensor0.Dim1};

  if (Tensor0.Dim1 == 0)
  {
    this->KrnlAddVec.setArg(0, Tensor0.Dim0);
    this->KrnlAddVec.setArg(1, Tensor0.Dim1);
    this->KrnlAddVec.setArg(2, Tensor0.Data);
    this->KrnlAddVec.setArg(3, Tensor1.Data);
    this->KrnlAddVec.setArg(4, Buf.Data);
    OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlAddVec, Events, &RunEvent));
  }
  else
  {
    this->KrnlAddMat.setArg(0, Tensor0.Dim0);
    this->KrnlAddMat.setArg(1, Tensor0.Dim1);
    this->KrnlAddMat.setArg(2, Tensor0.Data);
    this->KrnlAddMat.setArg(3, Tensor1.Data);
    this->KrnlAddMat.setArg(4, Buf.Data);
    OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlAddMat, Events, &RunEvent));
  }

  Events.push_back(RunEvent);

  return Buf;
};

template <typename DType>
TensorOnFPGA OCLWrap::Dot(TensorOnFPGA &Tensor0, TensorOnFPGA &Tensor1, std::vector<cl::Event> &Events)
{
  assert(Tensor0.Dim0 == Tensor1.Dim0 && Tensor0.Dim1 == Tensor1.Dim1);

  cl_int err;
  cl::Event RunEvent;
  TensorOnFPGA Buf = {this->AllocateReadWriteBuffer<DType>(Tensor0.Dim0 * Tensor0.Dim1), Tensor0.Dim0, Tensor0.Dim1};

  if (Tensor0.Dim1 == 0)
  {
    this->KrnlDotVec.setArg(0, Tensor0.Dim0);
    this->KrnlDotVec.setArg(1, Tensor0.Dim1);
    this->KrnlDotVec.setArg(2, Tensor0.Data);
    this->KrnlDotVec.setArg(3, Tensor1.Data);
    this->KrnlDotVec.setArg(4, Buf.Data);
    OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlDotVec, Events, &RunEvent));
  }
  else
  {
    this->KrnlDotMat.setArg(0, Tensor0.Dim0);
    this->KrnlDotMat.setArg(1, Tensor0.Dim1);
    this->KrnlDotMat.setArg(2, Tensor0.Data);
    this->KrnlDotMat.setArg(3, Tensor1.Data);
    this->KrnlDotMat.setArg(4, Buf.Data);
    OCL_CHECK(err, err = this->Queue.enqueueTask(this->KrnlDotMat, Events, &RunEvent));
  }

  Events.push_back(RunEvent);

  return Buf;
};

/**
 * Class AttentionLayer
 */

template <typename DType>
AttentionLayer::AttentionLayer()
{
}

template <typename DType>
AttentionLayer::~AttentionLayer()
{
}

template <typename DType>
TensorOnFPGA AttentionLayer::operator()(TensorOnFPGA Input)
{
  std::vector<TensorOnFPGA> Heads;

  for (unsigned int IterHead = 0; IterHead < HEAD_NUM; IterHead++)
  {
    TensorOnFPGA H0 = this->OCL.Mul(this->WQFPGA[IterHead].Data, Input);
    TensorOnFPGA H1 = this->OCL.Mul(this->WKFPGA[IterHead].Data, Input);
    TensorOnFPGA H2 = this->OCL.Mul(this->WVFPGA[IterHead].Data, Input);
    this->OCL.REmb(H0, H1, this->OCL.Freq);

    assert(this->KCache[IterHead].Dim0 + H1.Dim0 <= this->MaxCacheLen);
    assert(this->VCache[IterHead].Dim0 + H2.Dim0 <= this->MaxCacheLen);
    this->OCL.Append(this->KCache[IterHead], H1);
    this->OCL.Append(this->KCache[IterHead], H2);

    TensorOnFPGA H3 = this->OCL.Trans(H1);
    TensorOnFPGA H4 = this->OCL.Mul(H0, H3);
    TensorOnFPGA H5 = this->OCL.Softmax(H4);
    TensorOnFPGA H6 = this->OCL.Mul(H5, H2);

    H0.ReleaseMem();
    H1.ReleaseMem();
    H2.ReleaseMem();
    H3.ReleaseMem();
    H4.ReleaseMem();

    Heads.push_back(H6);
  }

  this->CurLen += Input.Dim0;
  
}

/**
 * Class FeedForwardLayer
 */

template <typename DType>
FeedForwardLayer::FeedForwardLayer()
{
}

template <typename DType>
FeedForwardLayer::~FeedForwardLayer()
{
}

template <typename DType>
TensorOnFPGA FeedForwardLayer::operator()(TensorOnFPGA Input)
{
  TensorOnFPGA H0 = this->OCL.Mul(this->W0FPGA, Input);
  TensorOnFPGA H1 = this->OCL.Silu(H0);
  TensorOnFPGA H2 = this->OCL.Mul(this->W2FPGA, Input);
  TensorOnFPGA H3 = this->OCL.Dot(H0, H2);
  TensorOnFPGA H4 = this->OCL.Mul(this->W1FPGA, H3);

  H0.ReleaseMem();
  H1.ReleaseMem();
  H2.ReleaseMem();
  H3.ReleaseMem();
  H4.ReleaseMem();

  return H4;
}

/**
 * Class TransformerBlock
 */

template <typename DType>
TransformerBlock::TransformerBlock()
{
}

template <typename DType>
TransformerBlock::~TransformerBlock()
{
}

template <typename DType>
TensorOnFPGA TransformerBlock::operator()(TensorOnFPGA Input)
{
  TensorOnFPGA H0 = this->NormAttn(Input);
  TensorOnFPGA H1 = this->Attn(H0);
  TensorOnFPGA H2 = this->OCL.Add(Input, H1);
  TensorOnFPGA H3 = this->NormFFN(H2);
  TensorOnFPGA H4 = this->FFN(H3);
  TensorOnFPGA H5 = this->OCL.Add(H2, H4);

  H0.ReleaseMem();
  H1.ReleaseMem();
  H2.ReleaseMem();
  H3.ReleaseMem();
  H4.ReleaseMem();

  return H5;
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

  GemmDispatch<int> GemmDispatchInstance(context, program, queue, 1);

  TensorOnHost<int> WHost = {GenerateRandomInput<int>(4096), EMBEDDING_DIM, 4096};
  TensorOnFPGA WFPGA;
  OCL_CHECK(err, WFPGA.Data = cl::Buffer(context, CL_MEM_READ_ONLY, (4096 * 4096) * sizeof(int), nullptr, &err));
  WFPGA.Dim0 = 4096;
  WFPGA.Dim1 = 4096;

  TensorOnHost<int> InputHost = {GenerateRandomInput<int>(16), EMBEDDING_DIM, 16};
  TensorOnFPGA InputFPGA;
  OCL_CHECK(err, InputFPGA.Data = cl::Buffer(context, CL_MEM_READ_ONLY, (4096 * 16) * sizeof(int), nullptr, &err));
  InputFPGA.Dim0 = 4096;
  InputFPGA.Dim1 = 16;

  TensorOnHost<int> ResHost;
  TensorOnFPGA ResFPGA;
  OCL_CHECK(err, ResFPGA.Data = cl::Buffer(context, CL_MEM_READ_ONLY, (4096 * 16) * sizeof(int), nullptr, &err));
  ResHost.Data = new int[4096 * 16];
  ResFPGA.Dim0 = 4096;
  ResFPGA.Dim1 = 16;

  OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({WFPGA.Data, InputFPGA.Data, ResFPGA.Data}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED));
  queue.finish();

  std::vector<cl::Event> Events;
  cl::Event e0, e1, e2;
  OCL_CHECK(err, err = queue.enqueueWriteBuffer(WFPGA.Data, CL_FALSE, 0, (WFPGA.Dim0 * WFPGA.Dim1) * sizeof(int), WHost.Data, nullptr, &e0));
  OCL_CHECK(err, err = queue.enqueueWriteBuffer(InputFPGA.Data, CL_FALSE, 0, (InputFPGA.Dim0 * InputFPGA.Dim1) * sizeof(int), InputHost.Data, nullptr, &e1));
  Events.push_back(e0);
  Events.push_back(e1);

  GemmDispatchInstance.request(WFPGA, InputFPGA, ResFPGA, Events);

  OCL_CHECK(err, err = queue.enqueueReadBuffer(ResFPGA.Data, CL_FALSE, 0, (ResFPGA.Dim0 * ResFPGA.Dim1) * sizeof(int), ResHost.Data, &Events, &e2));

  Events.push_back(e2);
  Events.back().wait();
  Events.clear();

  TensorOnHost<int> Res = WHost * InputHost;
  std::cout << ((Res == ResHost) ? "PASS!!!" : "ERROR!!!") << std::endl;
}
