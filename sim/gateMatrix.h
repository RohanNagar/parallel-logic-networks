
/*
  MATRIX INPUT to CUDA
  Each gate entry is 64 bits
  Output|  Gate| I1 row| I1 col| I0 row| I0 col
      63| 62-56|  55-42|  41-28|  27-14|   13-0
  Circuit gate width and height must be less than 16,384 because 14 bits per row and col
  Hardcoded, but can be modified for larger examples  
  Can be seen as a self contained unit for thread block
*/

#define DEBUG      1 // Debug checks

#define GATE_MASK  0x007F
#define INP_MASK   0x3FFF

#define GATE_SHFT  56
#define I0R_SHFT   14
#define I0C_SHFT    0
#define I1R_SHFT   42
#define I1C_SHFT   28

#define GATE_MASK  GATE_MASK << GATE_SHFT
#define I0R_MASK   INP_MASK << I0R_SHFT
#define I0C_MASK   INP_MASK << I0C_SHFT
#define I1R_MASK   INP_MASK << I1R_SHFT
#define I1C_MASK   INP_MASK << I1C_SHFT

#define GATE(x)    (x && GATE_MASK) << GATE_SHFT
#define I0R(x)     (x && INP_MASK) << I0R_SHFT
#define I0C(x)     (x && INP_MASK) << I0C_SHFT
#define I1R(x)     (x && INP_MASK) << I1R_SHFT
#define I1C(x)     (x && INP_MASK) << I1C_SHFT

enum GateType
{
  PORT_I,
  PORT_O,
  OBUF,
  RTL_INV,
  RTL_AND,
  RTL_OR,
  RTL_XOR,
  RTL_NAND,
  RTL_NOR,
  NUM_GATEs,
}

class gateMatrix{
  
  // creates matrix for CUDA for simulation 
  gateMatrix(uint32_t num_row, uint32_t num_col);
  
  // add a gate entry to the matrix
  void addGate(uint16_t O_row, uint16_t O_col, GateType gate, 
               uint16_t I0_row, uint16_t I0_col, uint16_t I1_row, uint16_t I1_col);
}
