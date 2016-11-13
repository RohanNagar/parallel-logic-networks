#include <stdint.h>
#include <string.h>
#include <stdlib.h> 
#include <iostream>
#include <fstream>
#include <vector>

#include "CudaMat.h" // using header to create the matrix
#include "gateMatrix.h" 
using namespace std;

void SimulateOnCuda(uint64_t** matrix);
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
  gateMatrix* matrix = createMatrixForCuda();

  SimulateOnCuda(CUDA_MATRIX, ADD_MATRIX_ROW, ADD_MATRIX_COL);
  
  matrix->printMatrix();

  delete matrix;
#endif
}

void SimulateOnCuda(uint64_t** matrix, uint32_t num_row, uint32_t num_col, 
                    uint64_t* input, uint64_t* output){
  // Initialize pointers for cuda memory
  uint64_t** d_matrix, d_input, d_output;
  uint32_t size = num_row * num_col * sizeof(uint64_t);
  
  // Allocate space for device copies
  cudaMalloc((void**)&d_matrix, size);
  
  // Copy inputs to device
  cudaMemcpy(d_matrix, matrix, size);
  
  // Launch kernel on CPU
  
  // Copy results back to host
  cudaMemcpy 

} 

#if DEBUG
gateMatrix* createMatrixForCuda(void){

  gateMatrix* matrix = new gateMatrix(CUDA_MATRIX_ROW, CUDA_MATRIX_COL);

  for(int i = 0; i < CUDA_MATRIX_ROW; i++){
    for(int j = 0; j < CUDA_MATRIX_COL; j++){
      matrix->addGate(CUDA_MATRIX[i][j], i, j);
    }
  } 
  return matrix;
}
#endif


