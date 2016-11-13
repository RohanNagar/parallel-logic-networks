#include <iostream>
#include <string>
#include <sstream>
#include "gateMatrix.h"
using namespace std;

int getI(string &input){
  int out;
 
  stringstream myStream(input);
  if(!(myStream >> out))
    while(1);
  return out;
}

int main(void){
  int Nr, Nc, Ni, No, Gt, Or, Oc, I0r, I0c, I1r, I1c;
  string input;
  bool done = false;

  cout << "Create gate list matrix for cuda \n";
  cout << "Enter number of gate levels: ";
  getline(cin, input); Nr = getI(input);
  cout << "Enter gate width per level: ";
  getline(cin, input); Nc = getI(input);
  cout << "Enter number of gate inputs: ";
  getline(cin, input); Ni = getI(input);
  cout << "Enter number of gate outputs: ";
  getline(cin, input); No = getI(input);
 
  gateMatrix manualMat = gateMatrix(Nr, Nc, Ni, No);
  
  while(!done){
    cout << "Enter gate type: ";
    getline(cin, input); Gt = getI(input);
    cout << "Enter gate level: ";
    getline(cin, input); Or = getI(input);
    cout << "Enter gate position: ";
    getline(cin, input); Oc = getI(input);
    cout << "Enter input0 gate level: ";
    getline(cin, input); I0r = getI(input);
    cout << "Enter input0 gate position: ";
    getline(cin, input); I0c = getI(input);
    cout << "Enter input1 gate level: ";
    getline(cin, input); I1r = getI(input);
    cout << "Enter input1 gate position: ";
    getline(cin, input); I1c = getI(input);

    manualMat.addGate(Or, Oc, (GateType)Gt, I0r, I0c, I1r, I1c);

    cout << "Done?";
    getline(cin, input);
    if(input[0] == 'Y')
      done = true;    
  }

  manualMat.printMatrix();
  manualMat.outputMatrixHeader();
}
