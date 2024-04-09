
// #include <stdio.h>
// #include <malloc.h>
// #include <stdlib.h>
// #include <string.h>
// #include <iostream>
// #include <vector>
// #include <chrono>

// #include "xcl2.hpp"
// #include "cmdlineparser.h"

// #define MATRIX_M 128
// #define MATRIX_N 128
// #define MATRIX_K 128

// using namespace sda;
// using namespace sda::utils;

// int main(int argc, char **argv)
// {

//   CmdLineParser parser;
//   parser.addSwitch("--fpga", "-x", "FPGA binary (xclbin) file to use");

//   parser.parse(argc, argv);
//   string fpgaBinary = parser.value("fpga");

//   if (fpgaBinary.size() == 0)
//   {
//     printf("ERROR: FPGA binary file (.xclbin) must be specified with the -x command line switch\n");
//     return -1;
//   }

//   printf("FPGA binary       : %s\n", fpgaBinary.c_str());

//   printf("Programming FPGA device\n");
//   cl_int err;
//   std::vector<cl::Device> devices = xcl::get_xil_devices();
//   devices.resize(1); // (arbitrarily) use the first Xilinx device that is found
//   OCL_CHECK(err, cl::Context context(devices[0], NULL, NULL, NULL, &err));
//   unsigned fileBufSize;
//   char *fileBuf = xcl::read_binary_file(fpgaBinary.c_str(), fileBufSize);
//   cl::Program::Binaries bins{{fileBuf, fileBufSize}};
//   OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
//   OCL_CHECK(err, cl::CommandQueue queue(context, devices[0], cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder, &err));

//   auto matrixA = new int[MATRIX_M * MATRIX_N];
//   auto matrixB = new int[MATRIX_N * MATRIX_K];
//   auto res = new int[MATRIX_M * MATRIX_K];
//   auto ref = new int[MATRIX_M * MATRIX_K];

//   for (int i = 0; i < MATRIX_M; i++)
//   {
//     for (int j = 0; j < MATRIX_N; j++)
//     {
//       matrixA[i * MATRIX_N + j] = rand();
//     }
//   }

//   for (int i = 0; i < MATRIX_N; i++)
//   {
//     for (int j = 0; j < MATRIX_K; j++)
//     {
//       matrixB[i * MATRIX_K + j] = rand();
//     }
//   }

//   for (int i = 0; i < MATRIX_M; i++)
//   {
//     for (int j = 0; j < MATRIX_K; j++)
//     {
//       int tmp = 0;
//       for (int k = 0; k < MATRIX_N; k++)
//         tmp += matrixA[i * MATRIX_N + k] * matrixB[k * MATRIX_K + j];
//       ref[i * MATRIX_K + j] = tmp;
//     }
//   }

//   printf("Running FPGA accelerator\n");

//   cl::Buffer matrixA_buf;
//   cl::Buffer matrixB_buf;
//   // cl::Buffer vector_buf;
//   cl::Buffer result_buf;
//   OCL_CHECK(err, matrixA_buf = cl::Buffer(context, CL_MEM_READ_ONLY, (MATRIX_M * MATRIX_N) * sizeof(int), nullptr, &err));
//   OCL_CHECK(err, matrixB_buf = cl::Buffer(context, CL_MEM_READ_ONLY, (MATRIX_N * MATRIX_K) * sizeof(int), nullptr, &err));
//   // OCL_CHECK(err, vector_buf = cl::Buffer(context, CL_MEM_READ_ONLY, (128) * sizeof(int), nullptr, &err));
//   OCL_CHECK(err, result_buf = cl::Buffer(context, CL_MEM_WRITE_ONLY, (MATRIX_M * MATRIX_K) * sizeof(int), nullptr, &err));

//   cl::Event in1_event;
//   cl::Event in2_event;
//   cl::Event run_event;
//   cl::Event out_event;
//   std::vector<cl::Event> events;

//   cl::Kernel kernel;
//   OCL_CHECK(err, kernel = cl::Kernel(program, "KrnlGemm", &err));
//   OCL_CHECK(err, err = kernel.setArg(0, MATRIX_M));
//   OCL_CHECK(err, err = kernel.setArg(1, MATRIX_N));
//   OCL_CHECK(err, err = kernel.setArg(2, MATRIX_K));
//   OCL_CHECK(err, err = kernel.setArg(3, matrixA_buf));
//   OCL_CHECK(err, err = kernel.setArg(4, matrixB_buf));
//   OCL_CHECK(err, err = kernel.setArg(5, result_buf));

//   size_t offset = 0;
//   OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({matrixA_buf, matrixB_buf, result_buf}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED));
//   queue.finish();

//   auto fpga_begin = std::chrono::high_resolution_clock::now();

//   OCL_CHECK(err, err = queue.enqueueWriteBuffer(matrixA_buf, CL_FALSE, offset, MATRIX_M * MATRIX_N * sizeof(int), matrixA, nullptr, &in1_event));
//   events.push_back(in1_event);
//   OCL_CHECK(err, err = queue.enqueueWriteBuffer(matrixB_buf, CL_FALSE, offset, MATRIX_N * MATRIX_K * sizeof(int), matrixB, nullptr, &in2_event));
//   events.push_back(in2_event);
//   OCL_CHECK(err, err = queue.enqueueTask(kernel, &events, &run_event));
//   events.push_back(run_event);
//   OCL_CHECK(err, err = queue.enqueueReadBuffer(result_buf, CL_FALSE, offset, MATRIX_M * MATRIX_K * sizeof(int), res, &events, &out_event));
//   events.push_back(out_event);

//   events.back().wait();
//   events.clear();

//   auto fpga_end = std::chrono::high_resolution_clock::now();

//   std::chrono::duration<double> fpga_duration = fpga_end - fpga_begin;

//   bool flag = 1;
//   for (int i = 0; i < MATRIX_M; i++)
//   {
//     for (int j = 0; j < MATRIX_K; j++)
//     {
//       if (ref[i * MATRIX_K + j] != res[i * MATRIX_K + j])
//         flag = 0;
//     }
//   }

//   std::cout << (flag ? "Test PASS !!!" : "ERROR !!!") << std::endl;

//   printf("FPGA Time         : %10.4f s\n", fpga_duration.count());
// }

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
  Res.Data = new DType[MA.Dim0 * MB.Dim1];
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
// FFNLayer::FFNLayer(){

// };

// template <typename DType>
// FFNLayer::FFNLayer(TensorOnHost<DType> WIn)
// {
//     this->WOnHost = WIn;
// }

// template <typename DType>
// void FFNLayer::migrate(cl::Context &Ctx, cl::CommandQueue &Queue)
// {
//     cl_int err;
//     OCL_CHECK(err, this->WOnFPGA = cl::Buffer(Ctx, CL_MEM_READ_ONLY, (this->WOnHost.Dim0 * this->WOnHost.Dim1) * sizeof(DType), nullptr, &err));
//     OCL_CHECK(err, err = Queue.enqueueMigrateMemObjects({this->WOnFPGA}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED));
//     Queue.finish();
// }

// template <typename DType>
// void FFNLayer::forward(TensorOnFPGA TokenInput, KrnlDispatch<GemmRequest<DType>> &Dispatch)
// {
//     Dispatch.request(this->WOnFPGA, TokenInput)
// }

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
void GemmRequest<DType>::run(TensorOnFPGA &MatrixA, TensorOnFPGA &MatrixB, TensorOnFPGA &Res, std::vector<cl::Event> &Events)
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

  OCL_CHECK(err, err = this->Q.enqueueTask(Kernel, &Events, &RunEvent));
  Events.push_back(RunEvent);

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
void GemmDispatch<DType>::request(TensorOnFPGA &MatrixA, TensorOnFPGA &MatrixB, TensorOnFPGA &Res, std::vector<cl::Event> &Events)
{
  this->Reqs[(this->Round++) % this->ReqNumMax].run(MatrixA, MatrixB, Res, Events);
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

  TensorOnHost<int> WHost = {GenerateRandomInput<int>(EMBEDDING_DIM), EMBEDDING_DIM, EMBEDDING_DIM};
  TensorOnFPGA WFPGA;
  OCL_CHECK(err, WFPGA.Data = cl::Buffer(context, CL_MEM_READ_ONLY, (EMBEDDING_DIM * EMBEDDING_DIM) * sizeof(int), nullptr, &err));
  WFPGA.Dim0 = EMBEDDING_DIM;
  WFPGA.Dim1 = EMBEDDING_DIM;

  TensorOnHost<int> InputHost = {GenerateRandomInput<int>(TEST_TOKEN_LEN), EMBEDDING_DIM, TEST_TOKEN_LEN};
  TensorOnFPGA InputFPGA;
  OCL_CHECK(err, InputFPGA.Data = cl::Buffer(context, CL_MEM_READ_ONLY, (EMBEDDING_DIM * TEST_TOKEN_LEN) * sizeof(int), nullptr, &err));
  InputFPGA.Dim0 = EMBEDDING_DIM;
  InputFPGA.Dim1 = TEST_TOKEN_LEN;

  TensorOnHost<int> ResHost;
  TensorOnFPGA ResFPGA;
  OCL_CHECK(err, ResFPGA.Data = cl::Buffer(context, CL_MEM_READ_ONLY, (EMBEDDING_DIM * TEST_TOKEN_LEN) * sizeof(int), nullptr, &err));
  ResHost.Data = new int[EMBEDDING_DIM * TEST_TOKEN_LEN];
  ResFPGA.Dim0 = EMBEDDING_DIM;
  ResFPGA.Dim1 = TEST_TOKEN_LEN;

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
