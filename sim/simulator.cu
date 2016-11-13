#include <stdint.h>
#include <string.h>
#include <stdlib.h> 
#include <iostream>
#include <fstream>
#include <vector>

#include "addMatrix.h" // using header to create the matrix 
using namepsace std;

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

  SimulateOnCuda(matrix);
#endif

  delete matrix;

}

void SimulateOnCuda(uint64_t** matrix){
  cudaMalloc

#if DEBUG
uint64_t** createMatrixForCuda(void){
  uint64_t** matrix = new uint64_t*[ADD_MATRIX_ROW];
  for(int i = 0; i < ADD_MATRIX_ROW; i++){
    matrix[i] = new uint64_t[ADD_MATRIX_COL];
    for(int j = 0; j < ADD_MATRIX_COL; j++){
      matrix[i] = ADD_MATRIX[i][j];
      cout << "" << ADD_MATRIX[i][j] << " ";
    }
    cout << "\n";
  } 
  return matrix;
}
#endif


