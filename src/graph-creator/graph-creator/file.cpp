
#include <iostream>
#include <fstream>
#include <sstream>

#include "file.h"
#include "graph.h"
#include "gate.h"

using std::cout;
using std::endl;

using std::ifstream;
using std::getline;

using std::stringstream;

using std::string;

namespace pln
{


file_parser::file_parser(std::string fname_in, std::string fname_out)
{
    m_filename_in = fname_in;
    m_filename_out = fname_out;
}

void file_parser::parse(graph& g)
{
    ifstream file_in;
    file_in.open(m_filename_in);

    if (!file_in.is_open())
    {
        cout << "Unable to open file " << m_filename_in << endl;
        return;
    }

    string cur_line;
    while (getline(file_in, cur_line))
    {
        // skip blank lines
        if (cur_line.length() == 0)
        {
            continue;
        }

        // skip beginning gates that are in our library
        stringstream ss{ cur_line };
        string first_token;
        ss >> first_token;

        if (gate::in_gate_lib(first_token))
        {
            cout << "in library" << endl;
            continue;
        }
        
        cout << first_token << endl;
    }
    while (file_in >> cur_line)
    {
        
        
        g.insert_gate(gate{});
    }

    file_in.close();
}


}




