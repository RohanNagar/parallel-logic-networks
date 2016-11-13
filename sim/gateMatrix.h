
#ifndef GATEMATRIX_H
#define GATEMATRIX_H

#include <stdint.h>
#include <string>
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

#define DEBUG      1 // Debug checks

// Number of bits for gate entry values
#define OUT_NUMB     2 // 0, 1, X, Z
#define GATE_NUMB    6  
#define I0R_NUMB    14
#define I0C_NUMB    14
#define I1R_NUMB    14
#define I1C_NUMB    14

// Bit mask for gate entry values
#define OUT_BMASK   0x0000000000000003
#define GATE_BMASK  0x000000000000002F
#define INP_BMASK   0x0000000000003FFF

// Location of gate entry values
#define OUT_SHFT    62
#define GATE_SHFT   56
#define I0R_SHFT    14
#define I0C_SHFT     0
#define I1R_SHFT    42
#define I1C_SHFT    28

// Mask for gate entry values
#define OUT_MASK    (OUT_BMASK << OUT_SHFT)
#define GATE_MASK   (GATE_BMASK << GATE_SHFT)
#define I0R_MASK    (INP_BMASK << I0R_SHFT)
#define I0C_MASK    (INP_BMASK << I0C_SHFT)
#define I1R_MASK    (INP_BMASK << I1R_SHFT)
#define I1C_MASK    (INP_BMASK << I1C_SHFT)

// Function macros to set gate entry values
#define setOUT(x)   (((uint64_t)x & OUT_BMASK) << OUT_SHFT)
#define setGATE(x)  (((uint64_t)x & GATE_BMASK) << GATE_SHFT)
#define setI0R(x)   (((uint64_t)x & INP_BMASK) << I0R_SHFT)
#define setI0C(x)   (((uint64_t)x & INP_BMASK) << I0C_SHFT)
#define setI1R(x)   (((uint64_t)x & INP_BMASK) << I1R_SHFT)
#define setI1C(x)   (((uint64_t)x & INP_BMASK) << I1C_SHFT)

// Function macros to get gate entry values
#define getOUT(x)   (((uint64_t)x >> OUT_SHFT) & OUT_BMASK)
#define getGATE(x)  (((uint64_t)x >> GATE_SHFT) & GATE_BMASK)
#define getI0R(x)   (((uint64_t)x >> I0R_SHFT) & INP_BMASK)
#define getI0C(x)   (((uint64_t)x >> I0C_SHFT) & INP_BMASK)
#define getI1R(x)   (((uint64_t)x >> I1R_SHFT) & INP_BMASK)
#define getI1C(x)   (((uint64_t)x >> I1C_SHFT) & INP_BMASK)

// Logic values
typedef enum LogicValue
{
  O,
  I,
  X,
  Z,  
}LogicValue;

// Gate type enum and names
typedef enum GateType
{
  NO_GATE,
  PORT_I,
  PORT_O,
  OBUF,
  RTL_INV,
  RTL_AND,
  RTL_OR,
  RTL_XOR,
  RTL_NAND,
  RTL_NOR,
  NUM_GATES,
}GateType;

string const GateNames[NUM_GATES] = {"___", " PI", " PO", "BUF", "INV", "AND", " OR", \
                                     "XOR", "NND", "NOR"};

// Class to create gate matrix
class gateMatrix{
private:
  uint64_t* matrix;
  uint32_t num_row;
  uint32_t num_col;
  uint32_t num_inp;
  uint32_t num_out;

public: 
// Constructor
  // creates matrix for CUDA for simulation 
  gateMatrix(uint32_t num_row, uint32_t num_col, uint32_t num_inp, uint32_t num_out);
  ~gateMatrix();

// Set and Get Functions

  // return raw matrix format
  uint64_t* getRawMatrix(void);
  uint32_t getNumRow(void);
  uint32_t getNumCol(void); 
  uint32_t getNumInp(void);
  uint32_t getNumOut(void);
  
  // add a gate entry to the matrix (either seperate values or entry itself)
  void addGate(uint16_t O_row, uint16_t O_col, GateType gate, 
               uint16_t I0_row, uint16_t I0_col, uint16_t I1_row, uint16_t I1_col);
  void addGate(uint64_t gate_entry, uint16_t O_row, uint16_t O_col);

#if DEBUG
  void clearGate(uint16_t O_row, uint16_t O_col);
  void printMatrix(void);
  void outputMatrixHeader(void);
//  void outputMatrixText(void);
#endif
};

#endif // GATEMATRIX_H
