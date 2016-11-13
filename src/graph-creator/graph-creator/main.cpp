
#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <map>
#include <unordered_map>

#include "file.h"
#include "vertex.h"

using std::cout;
using std::endl;

using std::ifstream;
using std::ofstream;

using std::string;

using std::map;
using std::unordered_map;

using pln::file_parser;

int main(int argc, char* argv[]) {

    // check arguments
    if (argc < 2) {
        cout << "Need to pass in an input file." << endl;
        exit(1);
        exit(2);
    }

    else if (argc > 2) {
        cout << "Too many arguments." << endl;
    }

    // open the file for parsing
    string const filename{ argv[1] };
    cout << "Filename is " << filename << endl;
    ifstream file_in;
    file_in.open(filename);
    
    std::unordered_map<string, string> map;
}
