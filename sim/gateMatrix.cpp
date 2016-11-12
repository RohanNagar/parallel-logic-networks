
#include <stdint.h>
#include "gateMatrix.h"
#if DEBUG
#include <iostream>
#include <fstream>
#include <iomanip>
#endif 

using namespace std;

/*
  MATRIX INPUT to CUDA
  Each gate entry is 64 bits
  Output|  Gate| I1 row| I1 col| I0 row| I0 col
      63| 62-56|  55-42|  41-28|  27-14|   13-0
  Circuit gate width and height must be less than 16,384 because 14 bits per row and col
  Hardcoded, but can be modified for larger examples  
  Can be seen as a self contained unit for thread block
*/

  gateMatrix::gateMatrix(uint32_t num_row, uint32_t num_col){
    matrix = new uint64_t*[num_row];
    for(int i = 0; i < num_row; i++)
      matrix[i] = new uint64_t[num_col];
    this->num_row = num_row;
    this->num_col = num_col;
  }

  gateMatrix::~gateMatrix(){
    for(int i = 0; i < num_row; i++)
      delete matrix[i];
    delete matrix;
  }

  void gateMatrix::addGate(uint16_t O_row, uint16_t O_col, GateType gate, 
                           uint16_t I0_row, uint16_t I0_col, uint16_t I1_row, uint16_t I1_col){
#if DEBUG
    if(num_row < O_row  || num_col < O_col  || 
       num_row < I0_row || num_col < I0_col || 
       num_row < I1_row || num_col < I1_col || 
       NUM_GATES < gate){
       cout << "Gate entry out of bounds [" << num_row << "][" << num_col << "]\n"; 
       while(1);
    }
#endif    
    matrix[O_row][O_col] = setGATE(gate) | setI0R(I0_row) | setI0C(I0_col) | setI1R(I1_row) | setI1C(I1_col);    
  }

#if DEBUG
  void gateMatrix::clearGate(uint16_t O_row, uint16_t O_col){
    if(num_row < O_row, num_col < O_col){
       cout << "Gate entry out of bounds\n"; 
    } else{
      matrix[O_row][O_col] = 0;
      cout << "Gate entry matrix[" << O_row << "][" << O_col << "] cleared\n";
    }
  }
  
  void gateMatrix::printMatrix(void){
    uint64_t value;
    for(int i = num_row - 1; i >= 0; i--){
      for(int j = 0; j < num_col; j++){
        value = matrix[i][j];
        cout << " " << GateNames[getGATE(value)] << "[" << getI0R(value) << "][" << getI0C(value) \
             << "] [" << getI1R(value) << "][" << getI1C(value) << "] "; 
      }
      cout << "\n";
    }
  }
  
  void gateMatrix::outputMatrixHeader(){
    ofstream outFile;
    outFile.open("CudaMat.h");
    outFile << "#include <stdint.h>\n\n";
    outFile << "#define CUDA_MATRIX_ROW " << num_row << "\n";
    outFile << "#define CUDA_MATRIX_COL " << num_col << "\n\n";
    outFile << "uint64_t const CUDA_MATRIX[" << num_row << "][" << num_col  << "] = { "; 
    for(int i = 0; i < num_row; i++){
      for(int j = 0; j < num_col; j++){
        outFile << "" << matrix[i][j] << ", ";
      }
      outFile << " \\\n"; 
    } 
    outFile << "};";
    outFile.close();
  }
  
/*  void gateMatrix::outputMatrixText(string &name){
    ofstream outFile;
    outFile.open(name += ".txt");*/
    
#endif 

