
#include <stdint.h>
#if DEBUG
#include <stdio.h>
#endif 
#include "gateMatrix.h"

/*
  MATRIX INPUT to CUDA
  Each gate entry is 64 bits
  Output|  Gate| I1 row| I1 col| I0 row| I0 col
      63| 62-56|  55-42|  41-28|  27-14|   13-0
  Circuit gate width and height must be less than 16,384 because 14 bits per row and col
  Hardcoded, but can be modified for larger examples  
  Can be seen as a self contained unit for thread block
*/

class gateMatrix{
private:
  uint32_t matrix**;
  uint32_t num_row;
  uint32_t num_col;
  
public:
  gateMatrix(uint32_t num_row, uint32_t num_col){
    matrix = new uint64_t[num_row][num_col];
  }

  ~gateMatrix(){
    delete matrix;
  }

  void addGate(uint16_t O_row, uint16_t O_col, GateType gate, 
               uint16_t I0_row, uint16_t I0_col, uint16_t I1_row, uint16_t I1_col){
#if DEBUG
    if(num_row < I0_row || num_col < I0_col || 
       num_row < I1_row || num_col < I1_col || 
       NUM_GATES < gate){
       cout << "Gate entry out of bounds"; 
       while(1);
    }
#endif    
    matrix[O_row][O_col] = GATE(gate) | I0R(I0_row) | I0C(I0_col) | I1R(I1_row) | I1C(I1_col);    
  }
}
