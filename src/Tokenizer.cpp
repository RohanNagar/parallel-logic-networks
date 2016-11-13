#include "Tokenizer.h"
#include <string.h>
#include <stdio.h>

Tokenizer::Tokenizer(char replaceWith, char delimiter){
  this->replaceWith = replaceWith;
  this->delimiter = delimiter;
}

string Tokenizer::replace(string *myString){
  string tString;
  
  for(int i = 0; i < myString->length(); i++)
  {
    tString[i] += (*myString)[i];     
printf("%s\n", tString.c_str());
    if((*myString)[i] == delimiter)
    {
      tString[i] = replaceWith;
    }
  }

  return tString;
}

struct Token{
  string** str;
  int num;
}; typedef struct Token Token;

string** Tokenizer::tokenize(string *myString)
{
  string** tString = new string*[256];
  int front = 0;
  int index = 0;

printf("%s\n", myString->c_str());
  for(int i = 0; i < myString->length(); i++)
  {
     while((*myString)[i] == delimiter)
     {
       i++;
       if(i == myString->length()) return tString;
     }
   
     front = i;      
   
     while((*myString)[i] != delimiter)
     {
       i++;
       if(i == myString->length()) break;
     }   
     tString[index] = new string(myString->substr(front, i- front));
     index++;
  }
  return tString;
}
