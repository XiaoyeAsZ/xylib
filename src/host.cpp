
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "xcl2.hpp"
#include "cmdlineparser.h"

#define MATRIX_M 128
#define MATRIX_N 128
#define MATRIX_K 128

using namespace sda;
using namespace sda::utils;

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
  devices.resize(1); // (arbitrarily) use the first Xilinx device that is found
  OCL_CHECK(err, cl::Context context(devices[0], NULL, NULL, NULL, &err));
  unsigned fileBufSize;
  char *fileBuf = xcl::read_binary_file(fpgaBinary.c_str(), fileBufSize);
  cl::Program::Binaries bins{{fileBuf, fileBufSize}};
  OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
  OCL_CHECK(err, cl::CommandQueue queue(context, devices[0], cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder, &err));

  auto matrixA = new int[MATRIX_M * MATRIX_N];
  auto matrixB = new int[MATRIX_N * MATRIX_K];
  auto res = new int[MATRIX_M * MATRIX_K];
  auto ref = new int[MATRIX_M * MATRIX_K];

  for (int i = 0; i < MATRIX_M; i++)
  {
    for (int j = 0; j < MATRIX_N; j++)
    {
      matrixA[i * MATRIX_N + j] = rand();
    }
  }

  for (int i = 0; i < MATRIX_N; i++)
  {
    for (int j = 0; j < MATRIX_K; j++)
    {
      matrixB[i * MATRIX_K + j] = rand();
    }
  }

  for (int i = 0; i < MATRIX_M; i++)
  {
    for (int j = 0; j < MATRIX_K; j++)
    {
      int tmp = 0;
      for (int k = 0; k < MATRIX_N; k++)
        tmp += matrixA[i * MATRIX_N + k] * matrixB[k * MATRIX_K + j];
      ref[i * MATRIX_K + j] = tmp;
    }
  }

  printf("Running FPGA accelerator\n");

  cl::Buffer matrixA_buf;
  cl::Buffer matrixB_buf;
  // cl::Buffer vector_buf;
  cl::Buffer result_buf;
  OCL_CHECK(err, matrixA_buf = cl::Buffer(context, CL_MEM_READ_ONLY, (MATRIX_M * MATRIX_N) * sizeof(int), nullptr, &err));
  OCL_CHECK(err, matrixB_buf = cl::Buffer(context, CL_MEM_READ_ONLY, (MATRIX_N * MATRIX_K) * sizeof(int), nullptr, &err));
  // OCL_CHECK(err, vector_buf = cl::Buffer(context, CL_MEM_READ_ONLY, (128) * sizeof(int), nullptr, &err));
  OCL_CHECK(err, result_buf = cl::Buffer(context, CL_MEM_WRITE_ONLY, (MATRIX_M * MATRIX_K) * sizeof(int), nullptr, &err));

  cl::Event in1_event;
  cl::Event in2_event;
  cl::Event run_event;
  cl::Event out_event;
  std::vector<cl::Event> events;

  cl::Kernel kernel;
  OCL_CHECK(err, kernel = cl::Kernel(program, "KrnlGemm", &err));
  OCL_CHECK(err, err = kernel.setArg(0, MATRIX_M));
  OCL_CHECK(err, err = kernel.setArg(1, MATRIX_N));
  OCL_CHECK(err, err = kernel.setArg(2, MATRIX_K));
  OCL_CHECK(err, err = kernel.setArg(3, matrixA_buf));
  OCL_CHECK(err, err = kernel.setArg(4, matrixB_buf));
  OCL_CHECK(err, err = kernel.setArg(5, result_buf));

  size_t offset = 0;
  OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({matrixA_buf, matrixB_buf, result_buf}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED));
  queue.finish();

  auto fpga_begin = std::chrono::high_resolution_clock::now();

  OCL_CHECK(err, err = queue.enqueueWriteBuffer(matrixA_buf, CL_FALSE, offset, MATRIX_M * MATRIX_N * sizeof(int), matrixA, nullptr, &in1_event));
  events.push_back(in1_event);
  OCL_CHECK(err, err = queue.enqueueWriteBuffer(matrixB_buf, CL_FALSE, offset, MATRIX_N * MATRIX_K * sizeof(int), matrixB, nullptr, &in2_event));
  events.push_back(in2_event);
  OCL_CHECK(err, err = queue.enqueueTask(kernel, &events, &run_event));
  events.push_back(run_event);
  OCL_CHECK(err, err = queue.enqueueReadBuffer(result_buf, CL_FALSE, offset, MATRIX_M * MATRIX_K * sizeof(int), res, &events, &out_event));
  events.push_back(out_event);

  events.back().wait();
  events.clear();

  auto fpga_end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> fpga_duration = fpga_end - fpga_begin;

  bool flag = 1;
  for (int i = 0; i < MATRIX_M; i++)
  {
    for (int j = 0; j < MATRIX_K; j++)
    {
      if (ref[i * MATRIX_K + j] != res[i * MATRIX_K + j])
        flag = 0;
    }
  }

  std::cout << (flag ? "Test PASS !!!" : "ERROR !!!") << std::endl;

  printf("FPGA Time         : %10.4f s\n", fpga_duration.count());
}
