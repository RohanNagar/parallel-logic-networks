#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
using std::string;

#ifndef TOKENIZER_H
#define TOKENIZER_H

class Tokenizer{
private:
  char replaceWith;
  char delimiter;
public:
  Tokenizer(char replaceWith, char delimiter);
  string** tokenize(string* myString);
  string replace(string* myString);
};

#endif
