#include <stdint.h>
#include <string.h>
#include <stdlib.h> 
#include <iostream>
#include <fstream>
#include <vector>

#include "CudaMat.h" // using header to create the matrix
#include "gateMatrix.h" 
using namespace std;


// when simulating with multiple inputs, try to not leave here..
// so setup a shared memory gate representation and work here... .
__global__ void Simulate(uint64_t* matrix, uint32_t num_row, uint32_t num_col, 
                         LogicValue* input, uint32_t num_inp,   
                         LogicValue* output, uint32_t num_out){ // use LogicValue** and num_tests
  extern __shared__ int sMatrix[];
  // int myId = threadIdx.x +blockDim.x * blockIdx.x;
  int tid = threadIdx.x; // TODO num_col == block? 

  // move gate network into shared memory
  for(int i = 0; i < num_row; i++){    
    sMatrix[i * num_col +  tid] =  matrix[i * num_col + tid];
    __syncthreads();
  }

  // enter input values 
  sMatrix[tid] = sMatrix[tid] |  
  
   


}
void SimulateOnCuda(gateMatrix* matrix, LogicValue* input, LogicValue* ouput);
#if DEBUG
gateMatrix* createMatrixForCuda(void);
#endif 

int main(void){  //int argc, char** argv){
  // currently using the header file instead of input file 
  // currently using hardcoded output file name

  // reserving space for function to create graph from python input text file (ben)
  
  // reserving space to create matrix from graph, gateMatrix.h (alvin)

  // Take graph matrix, and put it into cuda.... 
  // using for loop instead to create matrix from "addMatrix.h" (hard coded)
#if DEBUG
  LogicValue *input, *output;                   // given input and produced ouput
  gateMatrix* matrix = createMatrixForCuda();   // create matrix 
  input = new LogicValue[matrix->getNumInp()];  // create input 
  output = new LogicValue[matrix->getNumOut()]; // create output

  // give a test input hard coded
  input[0] = I;
  input[1] = I;
  output[0] = X;

  SimulateOnCuda(matrix, input, output);
  
  matrix->printMatrix();

  delete matrix;
#endif
}

void SimulateOnCuda(gateMatrix* matrix, LogicValue* input, LogicValue* output){
  // Initialize pointers for cuda memory
  uint64_t *d_matrix, *d_input, *d_output;
  uint32_t mat_size = matrix->getNumRow() * matrix->getNumCol() * sizeof(uint64_t);
  uint32_t inp_size = matrix->getNumInp() * sizeof(LogicValue);
  uint32_t out_size = matrix->getNumOut() * sizeof(LogicValue);

  // Allocate space for device copies
  cudaMalloc((void**)&d_matrix, mat_size);
  cudaMalloc((void**)&d_input, inp_size);
  cudaMalloc((void**)&d_output, out_size); 

  // Copy inputs to device
  cudaMemcpy(d_matrix, matrix->getRawMatrix(), mat_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, input, inp_size, cudaMemcpyHostToDevice);
  
  // Launch kernel on CPU
   
  // Copy results back to host
  cudaMemcpy(output, d_output, out_size, cudaMemcpyDeviceToHost);

} 

#if DEBUG
gateMatrix* createMatrixForCuda(void){

  gateMatrix* matrix = new gateMatrix(CUDA_MATRIX_ROW, CUDA_MATRIX_COL, 
                                      CUDA_MATRIX_INP, CUDA_MATRIX_OUT);

  for(int i = 0; i < CUDA_MATRIX_ROW; i++){
    for(int j = 0; j < CUDA_MATRIX_COL; j++){
      matrix->addGate(CUDA_MATRIX[i][j], i, j);
    }
  } 
  return matrix;
}
#endif


