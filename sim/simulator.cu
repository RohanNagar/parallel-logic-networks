#include <stdint.h>
#include <string.h>
#include <stdlib.h> 
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "Tokenizer.h"
#include "gateMatrix.h" 
using namespace std;

// GLobal functions 
void SimulateOnCuda(gateMatrix* matrix, LogicValue* input, LogicValue* ouput, uint32_t num_passes);
__global__ void Simulate(uint64_t* matrix, uint32_t num_row, uint32_t num_col, 
                         LogicValue* input, uint32_t num_inp,   
                         LogicValue* output, uint32_t num_out, uint32_t num_passes);
gateMatrix* createMatrixForCuda(void); 
void getInput(char* inputFile, gateMatrix* matrix, LogicValue* input, uint32_t* num_passes);
void printOutput(char* outputFile, gateMatrix* matrix, LogicValue* output, uint32_t num_passes);

// Files for input and output
char* inputFile = "input.txt";
char* outputFile = "output.txt";


int main(void){ 
  LogicValue *input, *output; // given inputs and produced ouputs
  uint32_t *num_passes;       // number of inputs passes to iterate over
 
  // reserving space to create matrix from graph, <Design.h>
  gateMatrix* matrix = createMatrixForCuda();
 
  // parse input file
  getInput(inputFile, matrix, input, num_passes);

  // simulate design
  SimulateOnCuda(matrix, input, output, *num_passes);

  // print output to file
  printOutput(outputFile, matrix, output, *num_passes);

  // deallocate memory
  delete input;
  delete output;
  delete matrix;
}



// Initialize Memory to simulate for Cuda
void SimulateOnCuda(gateMatrix* matrix, LogicValue* input, LogicValue* output, uint32_t num_passes){
  // Initialize pointers for cuda memory
  uint64_t *d_matrix;
  LogicValue *d_input, *d_output;
  uint32_t mat_size = matrix->getNumRow() * matrix->getNumCol() * sizeof(uint64_t);
  uint32_t inp_size = matrix->getNumInp() * num_passes * sizeof(LogicValue);
  uint32_t out_size = matrix->getNumOut() * num_passes * sizeof(LogicValue);

  // Allocate space for device copies
  cudaMalloc((void**)&d_matrix, mat_size);
  cudaMalloc((void**)&d_input, inp_size);
  cudaMalloc((void**)&d_output, out_size);

  // Copy matrix and inputs to device
  cudaMemcpy(d_matrix, matrix->getRawMatrix(), mat_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, input, inp_size, cudaMemcpyHostToDevice); 

#ifdef PRINTALL
  //int iterations = num_passes;
  //num_passes = 1;
  //for(int i = 0; i < iterations; i++){
#endif    
  // Launch Kernel on GPU
  Simulate<<<1, matrix->getNumCol(), mat_size>>>(d_matrix, matrix->getNumRow(), matrix->getNumCol(),
                                                 d_input, inp_size, d_output, out_size, num_passes);

  // Copy matrix and results back to host
  cudaMemcpy(matrix->getRawMatrix(), d_matrix, mat_size, cudaMemcpyDeviceToHost); 
  cudaMemcpy(output, d_output, out_size, cudaMemcpyDeviceToHost);

#ifdef PRINTALL
  //}
#endif
} 

 
// when simulating with multiple inputs, try to not leave here..
// so setup a shared memory gate representation and work here... .
__global__ void Simulate(uint64_t* matrix, uint32_t num_row, uint32_t num_col, 
                         LogicValue* input, uint32_t num_inp,   
                         LogicValue* output, uint32_t num_out, uint32_t num_passes){ 
  extern __shared__ uint64_t sMatrix[];
  // int myId = threadIdx.x +blockDim.x * blockIdx.x;
  uint32_t tid = threadIdx.x; // TODO num_col == block? 
  uint64_t gateEntry;
  int gateInp0, gateInp1, gateOut;

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
    gateInp0  = (LogicValue)getOUT(sMatrix[getI0R(gateEntry) * num_col + getI0C(gateEntry)]); 
    gateInp1  = (LogicValue)getOUT(sMatrix[getI1R(gateEntry) * num_col + getI1C(gateEntry)]);

    // TODO find a way to simplify?
    switch(getGATE(gateEntry)){
      case NO_GATE:
        gateOut = 0;
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
        gateOut = gateInp0 & gateInp1;
        break;
      case RTL_OR:
        gateOut = gateInp0 | gateInp1;
        break;
      case RTL_XOR: // only works for 0 and 1
        gateOut = gateInp0 ^ gateInp1;
        break;
      case RTL_NAND:
        gateOut = !(gateInp0 & gateInp1);
        break;
      case RTL_NOR:
        gateOut = !(gateInp0 | gateInp1);
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

  // enter output values 
  if(tid < num_out){
    output[tid] = (LogicValue)setOUT(sMatrix[(num_row - 1) * num_col + tid]);
  }
}


/* HELPER FUNCTIONS */

// create Matrix for Cuda from Design header file (DESIGN.h) 
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

void getInput(char* inputFile, gateMatrix* matrix, LogicValue* input, uint32_t* num_passes){
  ifstream file;
  file.open(inputFile);
  if(!file){
    cout << "Error: Can't open the file.\n";
    exit(1); 
  }
  
  string str; const char* delim = " ";

  // get number of passes
  if(getline(file,str)){
    char* str_c = strdup(str.c_str());
    char* token = strtok(str_c, delim);
    *num_passes = atoi(token);
cout << "" << *num_passes;
    free(str_c);
  }
  
  input = new LogicValue[matrix->getNumInp() *  (*num_passes)];

  // get inputs
  for(int i = 0; i < *num_passes; i++){
    int j = 0;
    while(getline(file,str)){
      char* str_c = strdup(str.c_str());
      char* token = strtok(str_c, delim);
      input[i * matrix->getNumInp() + j] = (LogicValue)atoi(token);
cout << "" << input[i * matrix->getNumInp() + j];
     j++;
    }
  }

  file.close();
  exit(1);
}

void printOutput(char* outputFile, gateMatrix* matrix, LogicValue* output, uint32_t num_passes){

}


/*  Tokenizer T = Tokenizer(' ', ' ');

  int getI(string &input){
    int out;
    stringstream myStream(input);
    if(!(myStream >> out))
      cout << "getI broken";
    return out;
  } 

  LogicValue* getUserInput(gateMatrix* matrix){
    string in; string** token;
    LogicValue* input = new LogicValue[matrix->getNumInp()];  
 
    cout << "Enter input of size " << matrix->getNumInp() << ". (ex: 0 1 0 0 1)";
    getline(cin, in); 
    token = T.tokenize(&in);
    for(int i = 0; i < matrix->getNumInp(); i++){
      input[i] = (LogicValue)getI(*token[i]); 
      cout << "" << input[i];
    }
    return input; 
  }
  //printUserOutput();
*/
