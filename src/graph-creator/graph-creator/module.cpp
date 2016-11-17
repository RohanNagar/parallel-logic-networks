
#include "module.h"

using std::string;
using std::vector;

using std::unordered_map;

namespace pln
{

module module::module_err{ "ERR" };

module::module(string const & name) :
    m_name{ name },
    m_input_ports{ vector<gid_t>{} },
    m_output_ports{ vector<gid_t>{} },
    m_gate_list{ unordered_map<string, gid_t>{} }
{

}


string const & module::get_name() const
{
    return m_name;
}

void module::insert_input_port(gate const & port)
{
    m_input_ports.push_back(port.get_id());
    insert_gate(port);
}


void module::insert_output_port(gate const & port)
{
    m_output_ports.push_back(port.get_id());
    insert_gate(port);
}

void module::insert_internal_module(string const & name, string const & mod_name)
{   // name is the local name to the outer module and mod_name is the actual module's name
    m_module_list.insert({ name, mod_name });
}

void module::insert_gate(gate const & gt)
{
    m_gate_list.insert({ gt.get_name(), gt.get_id() });
}


gid_t module::find_gate(string const & name)
{
    auto it = m_gate_list.find(name);
    if (it == m_gate_list.end())
    {
        return -1;
    }
    return it->second;
}

string const & module::find_internal_module(string const & name)
{
    auto it = m_module_list.find(name);
    if (it == m_module_list.end())
    {
        return "";
    }
    return it->second;
}


}


