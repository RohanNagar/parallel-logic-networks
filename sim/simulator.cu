#include <stdint.h>
#include <string.h>
#include <stdlib.h> 
#include <iostream>
#include <fstream>
#include <vector>

#include "addMatrix.h" // using header to create the matrix 
using namespace std;

void SimulateOnCuda(uint64_t** matrix);

int main(void){  //int argc, char** argv){
  // currently using the header file instead of input file 
  // currently using hardcoded output file name

  // reserving space for function to create graph from python input text file (ben)
  
  // reserving space to create matrix from graph, gateMatrix.h (alvin)

  // Take graph matrix, and put it into cuda.... 
  // using for loop instead to create matrix from "addMatrix.h" (hard coded)
#if DEBUG
  uint64_t** matrix = createMatrixForCuda();

//  SimulateOnCuda(CUDA_MATRIX, ADD_MATRIX_ROW, ADD_MATRIX_COL);

  delete matrix;
#endif
}
/*
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

} */

#if DEBUG
uint64_t** createMatrixForCuda(void){
  uint64_t** matrix = new uint64_t*[CUDA_MATRIX_ROW];
  for(int i = 0; i < CUDA_MATRIX_ROW; i++){
    matrix[i] = new uint64_t[CUDA_MATRIX_COL];
    for(int j = 0; j < CUDA_MATRIX_COL; j++){
      matrix[i] = CUDA_MATRIX[i][j];
      cout << "" << CUDA_MATRIX[i][j] << " ";
    }
    cout << "\n";
  } 
  return matrix;
}
#endif


