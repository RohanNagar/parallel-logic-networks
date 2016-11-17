#include <stdint.h>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip> 
#include "gateMatrix.h"
using namespace std;

/*
  MATRIX INPUT to CUDA
  Each gate entry is 64 bits
  Output|  Gate| I1 row| I1 col| I0 row| I0 col
   63-62| 61-56|  55-42|  41-28|  27-14|   13-0
  Circuit gate width and height must be less than 16,384 because 14 bits per row and col
  Hardcoded, but can be modified for larger examples  
  Can be seen as a self contained unit for thread block
*/

  // Construtor 
  gateMatrix::gateMatrix(uint32_t num_row, uint32_t num_col, uint32_t num_inp, uint32_t num_out){
    matrix = new uint64_t[num_row * num_col]();
    this->num_row = num_row;
    this->num_col = num_col;
    this->num_inp = num_inp;
    this->num_out = num_out;
  }

  gateMatrix::~gateMatrix(){
    delete[] matrix;
  }

  // Set and Get Functions
  uint64_t* gateMatrix::getRawMatrix(void){
    return matrix;
  }
  
  uint32_t gateMatrix::getNumRow(void){
    return num_row;
  }

  uint32_t gateMatrix::getNumCol(void){
    return num_col;
  }

  uint32_t gateMatrix::getNumInp(void){
    return num_inp;
  }

  uint32_t gateMatrix::getNumOut(void){
    return num_out;
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
    matrix[O_row * num_col + O_col] = setGATE(gate) | setI0R(I0_row) | setI0C(I0_col) | 
                                                      setI1R(I1_row) | setI1C(I1_col);    
  }

  void gateMatrix::addGate(uint64_t gate_entry, uint16_t O_row, uint16_t O_col){
    matrix[O_row * num_col + O_col] = gate_entry;
  }

  void gateMatrix::clearGate(uint16_t O_row, uint16_t O_col){
    if(num_row < O_row || num_col < O_col){
       cout << "Gate entry out of bounds\n"; 
    } else{
      matrix[O_row * num_col + O_col] = 0;
      cout << "Gate entry matrix[" << O_row << "][" << O_col << "] cleared\n";
    }
  }
  
  void gateMatrix::printMatrix(void){
    uint64_t value;
    for(int i = num_row - 1; i >= 0; i--){
      for(int j = 0; j < num_col; j++){
        value = matrix[i * num_col + j];
        cout << "" << LogicNames[getOUT(value)] <<  " " << GateNames[getGATE(value)] \
             << "[" << getI0R(value) << "][" << getI0C(value) \
             << "] [" << getI1R(value) << "][" << getI1C(value) << "] "; 
      }
      cout << "\n";
    }
  }
  
  void gateMatrix::outputMatrixHeader(char* name){
    ofstream outFile;
    outFile.open(name);
    outFile << "#include <stdint.h>\n\n";
    outFile << "#define CUDA_MATRIX_ROW " << num_row << "\n";
    outFile << "#define CUDA_MATRIX_COL " << num_col << "\n";
    outFile << "#define CUDA_MATRIX_INP " << num_inp << "\n";
    outFile << "#define CUDA_MATRIX_OUT " << num_out << "\n\n";
    outFile << "uint64_t const CUDA_MATRIX[" << std::setw(2) << std::setfill('0') << num_row << "]["
<< std::setw(2) << std::setfill('0') << num_col  << "] = { "; 
    for(int i = 0; i < num_row; i++){
      for(int j = 0; j < num_col; j++){
        outFile << "" << matrix[i * num_col + j] << ", ";
      }
      outFile << " \n"; 
    } 
    outFile << "};";
    outFile.close();
  }
  
/*  void gateMatrix::outputMatrixText(string &name){
    ofstream outFile;
    outFile.open(name += ".txt");*/
    

