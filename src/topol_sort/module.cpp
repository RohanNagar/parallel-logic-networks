
#include "module.h"

using std::string;
using std::vector;

using std::unordered_map;

namespace pln
{


module::module(string const & name) :
    m_name{ name },
    m_input_ports{ vector<gtid_t>{} },
    m_output_ports{ vector<gtid_t>{} },
    m_gate_list{ unordered_map<string, gtid_t>{} }
{

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

void module::insert_gate(gate const & gt)
{
    m_gate_list.insert({ gt.get_name(), gt.get_id() });
}


gtid_t module::find_gate(string const & name)
{
    auto it = m_gate_list.find(name);
    if (it == m_gate_list.end())
    {
        return -1;
    }
    return it->second;
}


}


