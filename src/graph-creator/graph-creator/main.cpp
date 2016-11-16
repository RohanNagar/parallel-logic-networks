
#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <map>
#include <unordered_map>

#include "file.h"
#include "gate.h"
#include "graph.h"

using std::cout;
using std::endl;

using std::ifstream;
using std::ofstream;

using std::string;

using namespace pln;


const string OUT_FILENAME = "graph.txt";

int main(int argc, char* argv[])
{

    // check arguments
    if (argc < 2)
    {
        cout << "Need to pass in an input file." << endl;
        exit(1);
    }

    else if (argc > 2)
    {
        cout << "Too many arguments." << endl;
        exit(1);
    }

    // open the file for parsing
    string const in_filename{ argv[1] };
    string const out_filename{ OUT_FILENAME };
    cout << "Filename is " << in_filename << endl;

    file_parser fp{ in_filename, out_filename };

    graph g{};
    
    fp.parse(g);

    g.print();

}

