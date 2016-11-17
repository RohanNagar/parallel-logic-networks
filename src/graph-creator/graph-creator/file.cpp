
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
    // file_in.open(m_filename_in);
    file_in.open("sub_4bit_edn.txt");

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
    while (file_in)
    {
        stringstream ss{ cur_line };
        string cur_token;
        ss >> cur_token;
        
        module new_module{ cur_token };
        cout << "New module: " << cur_token << endl;

        // create the module inputs and outputs
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
            gate in_gate{ string{ cur_token }, "PORT_I" };
            cout << "Added input port " << cur_token << endl;
            g.insert_gate(in_gate);
            new_module.insert_input_port(in_gate);

            ss >> cur_token;
        }

        // move to the first output port
        while (ss >> cur_token)
        {
            gate out_gate{ cur_token, "PORT_O" };
            cout << "Added output port " << cur_token << endl;
            g.insert_gate(out_gate);
            new_module.insert_output_port(out_gate);
        }


        /* Parse Instances */

        // make sure instances is the next keyword
        getline(file_in, cur_line);
        ss = stringstream{ cur_line };
        ss >> cur_token;
        if (cur_token != "Instances:")
        {
            cout << "Syntax error: expected Instances keyword" << endl;
            return;
        }

        // create all the instances, which could be modules or gates
        while (getline(file_in, cur_line))
        {
            if (file_in.fail())
            {
                cout << "Syntax error: expected Nets keyword." << endl;
                return;
            }

            // the first token is the name, and the second token is the module/keyword
            string name;
            string instance;
            ss = stringstream{ cur_line };
            ss >> name >> instance;
            if (name == "Nets:")
            {
                break;
            }
            cout << "name: " << name << ", instance: " << instance << endl;
            if (gate::in_gate_lib(instance))
            {
                gate new_gate{ name, instance };
                g.insert_gate(new_gate);
                new_module.insert_gate(new_gate);
            }
            else
            {
                cout << "Instance is a module, not a gate." << endl;
            }
        }


        /* Parse Nets */
        // create all the nets
        while (getline(file_in, cur_line))
        {
            if (cur_line.size() == 0)
            {
                cout << "End of module." << endl;
                break;
            }
            ss = stringstream{ cur_line };
            string src;
            string dest;
            string port;
            gid_t gid_src;
            gid_t gid_dest;
            ss >> dest >> string{};

            while (ss >> src)
            {
                cout << "src: " << src << ", dest: " << dest << endl;

                // check if dest is a module
                module const & mod = g.find_module(dest);
                cout << mod.get_name() << endl;
                if (&mod != &module::module_err)
                {   // is a module
                    cout << "Is a module." << endl;
                }
                else
                {   // just a normal gate
                    gid_dest = new_module.find_gate(dest);
                    if (gid_dest == -1)
                    {
                        cout << "dest not found." << endl;
                        return;
                    }
                }
                

                gid_src = new_module.find_gate(src);

                if (gid_src == -1)
                {
                    cout << "src not found." << endl;
                    return;
                }
                
                g.insert_edge(gid_src, gid_dest);
                if (ss >> port && port[port.length() - 1] == ',')
                {
                    // cout << "More than one connection." << endl;
                }
                else
                {
                    // cout << "Done with line." << endl;
                }
            }
        }

        // add module to the graph
        g.insert_module(new_module);
        g.print();
        getline(file_in, cur_line);
    }

    file_in.close();
}


}




