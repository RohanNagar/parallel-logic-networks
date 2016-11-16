
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


file_parser::file_parser(string const & fname_in, string const & fname_out) :
    m_filename_in{ fname_in },
    m_filename_out{ fname_out }
{

}

void file_parser::parse(graph & g)
{
    ifstream file_in;
    file_in.open("add_edn.txt");

    if (!file_in.is_open())
    {
        cout << "Unable to open file " << m_filename_in << endl;
        return;
    }

    string cur_line;
    // skip beginning gates that are in our library
    while (getline(file_in, cur_line))
    {
        // skip blank lines
        if (cur_line.length() == 0)
        {
            continue;
        }

        stringstream ss{ cur_line };
        string first_token;
        ss >> first_token;

        if (!gate::in_gate_lib(first_token))
        {
            break;
        }

        cout << first_token << " in library" << endl;
    }
    
    // parse all the modules
    // at this point, cur_line contains the first line of a new module
    while (1)
    {
        stringstream ss{ cur_line };
        string cur_token;
        ss >> cur_token;
        
        module new_module{ cur_token };
        cout << "New module name is " << cur_token << endl;

        ss >> cur_token;
        if (cur_token != "-i")
        {
            cout << "Syntax error: expected -i after module name." << endl;
            return;
        }

        ss >> cur_token;            // move to the first input port
        while (cur_token != "-o")
        {
            if (ss.fail())
            {
                cout << "Syntax error: expected -o after inputs." << endl;
                return;
            }
            gate in_gate{ cur_token, "PORT_I" };
            cout << "Added input port " << cur_token << endl;
            g.insert_gate(in_gate);
            new_module.add_input_port(in_gate.get_id());

            ss >> cur_token;
        }

        ss >> cur_token;            // move to the first output port
        while (ss)
        {
            gate out_gate{ cur_token, "PORT_O" };
            cout << "Added output port " << cur_token << endl;
            g.insert_gate(out_gate);
            new_module.add_output_port(out_gate.get_id());

            ss >> cur_token;
        }

        getline(file_in, cur_line);
        break;
    }

    file_in.close();
}


}




