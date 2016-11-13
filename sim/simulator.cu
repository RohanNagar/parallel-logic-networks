#include <stdint.h>
#include <string.h>
#include <stdlib.h> 
#include <iostream>
#include <fstream>
#include <vector>

#include "CudaMat.h" // using header to create the matrix
#include "gateMatrix.h" 
using namespace std;

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
  uint32_t size = matrix->getNumRow() * matrix->getNumCol() * sizeof(uint64_t);
   
  // Allocate space for device copies
  cudaMalloc((void**)&d_matrix, size);
  cudaMalloc((void**)&d_input, matrix.getNumInp*sizeof(LogicValue));
  cudaMalloc((void**)&d_output, matrix.getNumOut*sizeof(LogicValue));

  // Copy inputs to device
 cudaMemcpy(d_matrix, matrix->getRawMatrix(), size);
  
  // Launch kernel on CPU
  
  // Copy results back to host
//  cudaMemcpy 

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


