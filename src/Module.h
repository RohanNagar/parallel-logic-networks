#include <string>
#include "Tokenizer.h"

using namespace std;


class Module{
  String name;
  List<*Port> inputs; 
  List<*Port> outputs;
  Tokenizer tokenize;
 
  Module(String *IO){
    &tokenize = new Tokenizer(' ', ' ');
    string** tokens = tokenize.tokenize(IO);

    &name = new string(tokens[0]);

    &inputs = new List<>();
    inputs.add
  }

}

class Gate:public Module{

}

class Port:public Module{
  Port(String* name){
    this.name = *new string(name);
    this.inputs = NULL;
    this
  }
}
