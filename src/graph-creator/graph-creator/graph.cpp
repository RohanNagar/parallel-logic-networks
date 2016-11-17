
#include <iostream>

#include "graph.h"

using std::cout;
using std::endl;

using std::string;
using std::vector;

namespace pln
{

/**
* Create an empty ordered map.
*/
graph::graph() :
    m_graph{ std::vector<std::vector<gid_t>>{} },
    m_gate_list{ std::vector<gate>{} }
{

}

void graph::insert_gate(gate const & new_gate)
{
    std::vector<gid_t> adj_vertices{};
    m_graph.push_back(adj_vertices);

    // map the new gate to its index
    // cout << "Inserting " << new_gate.get_name() << endl;
    m_gate_list.push_back(new_gate);
}

void graph::insert_edge(gid_t src, gid_t dest)
{
    if (src < 0 || src > m_graph.size())
    {
        cout << "Gate " << src << " does not exist." << endl;
        exit(1);
    }
    if (dest < 0 || dest > m_graph.size())
    {
        cout << "Gate " << dest << " does not exist." << endl;
        exit(1);
    }
    m_graph[src].push_back(dest);
}

void graph::insert_module(module const & mod)
{
    m_module_list.push_back(mod);
}

module const & graph::find_module(string const & name)
{
    cout << "searching for module " << name << endl;
    for (uint32_t i = 0; i < m_module_list.size(); ++i)
    {
        if (m_module_list[i].get_name() == name)
        {
            return m_module_list[i];
        }
    }
    return module::module_err;
}

void graph::set_heights(std::vector<gid_t> const & start_vertices)
{

}

void graph::print()
{
    cout << "Graph information:" << endl << endl;

    cout << "Adjacency List:" << endl;
    const char separator = '\t';
    const uint8_t width_id = 3;
    const uint8_t width_name = 10;
    const uint8_t width_type = 6;
    cout << "ID\t(Name)\t\t(Type)\t\tAdjacent Gates(ID)" << endl;
    for (uint32_t i = 0; i < m_graph.size(); ++i)
    {
        gate cur_gate = m_gate_list[i];
        vector<gid_t> adj_gates = m_graph[i];
        cout << cur_gate.get_id() << '\t' <<
            '(' << cur_gate.get_name() << ')' << "\t\t" <<
            '(' << cur_gate.get_type() << ')' << "\t\t";
        if (adj_gates.size() == 2)
        {
            cout << adj_gates[0] << ", " << adj_gates[1];
        }
        else if (adj_gates.size() == 1)
        {
            cout << adj_gates[0];
        }
        cout << endl;
    }
    cout << endl;

    cout << "Graph modules:" << endl;
    for (uint32_t i = 0; i < m_module_list.size(); ++i)
    {
        cout << m_module_list[i].get_name();
        if (i != m_module_list.size() - 1)
        {
            cout << ", ";
        }
    }
    cout << endl << endl;
}



}



