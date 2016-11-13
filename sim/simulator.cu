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
  extern __shared__ uint64_t sMatrix[];
  // int myId = threadIdx.x +blockDim.x * blockIdx.x;
  uint32_t tid = threadIdx.x; // TODO num_col == block? 
  uint32_t gateEntry, gateInp0, gateInp1, gateOut;

  // move gate network into shared memory
  for(uint32_t i = 0; i < num_row; i++){    
    sMatrix[i * num_col +  tid] =  matrix[i * num_col + tid];
    __syncthreads();
  }

  // enter input values (0) 
  if(tid < num_inp){
    sMatrix[tid] &= (~OUT_MASK);
    sMatrix[tid] |= setOUT(input[tid]); // TODO will need to fix based on location of input..
    __syncthreads();
  } 

  // evaluate circuit (0 -> num_row - 1)
  for(uint32_t i = 1; i < num_row; i++){
    gateEntry = sMatrix[i * num_col + tid];    
    gateInp0  = getOUT(sMatrix[getI0R(gateEntry) * num_col + getI0C(gateEntry)]); 
    gateInp1  = getOUT(sMatrix[getI1R(gateEntry) * num_col + getI1C(gateEntry)]);

    // TODO find a way to simplify?
    switch(getGATE(gateEntry)){
      case NO_GATE:
        gateOut = X;
        break;
      case PORT_I:
        break;
      case PORT_O:
      case OBUF:
        gateOut = gateInp0; 
        break;
      case RTL_INV: // TODO for all gates
        switch(gateInp0){
          case O:
            gateOut = I;
            break;
          case I: 
            gateOut = O;
            break;
          case X:
            gateOut = X;
            break;
          case Z:
            gateOut = Z;
            break;
        }
        break;
      case RTL_AND:
        gateOut = gateInp0 * gateInp1;
        break;
      case RTL_OR:
        gateOut = gateInp0 + gateInp1;
        break;
      case RTL_XOR:
        gateOut = gateInp0 ^ gateInp1;
        break;
      case RTL_NAND:
        gateOut = !(gateInp0 * gateInp1);
        break;
      case RTL_NOR:
        gateOut = !(gateInp0 + gateInp1);
        break;
      default:
        break;
    }
    sMatrix[i * num_col + tid] &= (~OUT_MASK);
    sMatrix[i * num_col + tid] |= setOUT(gateOut);
    __syncthreads(); 
  } 

  // test code
   for(uint32_t i = 0; i < num_row; i++){    
    matrix[i * num_col +  tid] =  sMatrix[i * num_col + tid];
    __syncthreads();
  } 
  return;

  // enter output values 
  if(tid < num_out){
    output[tid] = (LogicValue)setOUT(sMatrix[(num_row - 1) * num_col + tid]);
  }
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
  cout << "Inputs: " << input[0] << " " << input[1];
  cout << "\nOutputs: " << output[0];

  delete matrix;
#endif
}

void SimulateOnCuda(gateMatrix* matrix, LogicValue* input, LogicValue* output){
  // Initialize pointers for cuda memory
  uint64_t *d_matrix;
  LogicValue *d_input, *d_output;
  uint32_t mat_size = matrix->getNumRow() * matrix->getNumCol() * sizeof(uint64_t);
  uint32_t inp_size = matrix->getNumInp() * sizeof(LogicValue);
  uint32_t out_size = matrix->getNumOut() * sizeof(LogicValue);

  // Allocate space for device copies
  cudaMalloc((void**)&d_matrix, mat_size);
  cudaMalloc((void**)&d_input, inp_size);
  cudaMalloc((void**)&d_output, out_size); 
  cout << "Allocating Space\n";

  // Copy inputs to device
  cudaMemcpy(d_matrix, matrix->getRawMatrix(), mat_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, input, inp_size, cudaMemcpyHostToDevice);
  cout << "Copying Inputs \n";  

  // Launch kernel on CPU
  Simulate<<<1, matrix->getNumCol(), mat_size>>>(d_matrix, matrix->getNumRow(), matrix->getNumCol(),
                                                 d_input, inp_size, d_output, out_size);
  cout << "Completed simulation \n";
  
  // test code 
  cudaMemcpy(matrix->getRawMatrix(), d_matrix, mat_size, cudaMemcpyDeviceToHost); 
  return;
 
  // Copy results back to host
  cudaMemcpy(output, d_output, out_size, cudaMemcpyDeviceToHost);
  cout << "Copying Outputs\n";
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


