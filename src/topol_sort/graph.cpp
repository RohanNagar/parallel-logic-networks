
#include <iostream>

#include "graph.h"

using std::cout;
using std::endl;

using std::vector;

namespace pln
{

/**
* Create an empty ordered map.
*/
graph::graph() :
    m_graph{ std::vector<std::vector<gtid_t>>{} },
    m_gate_list{ std::vector<gate>{} }
{

}

// ALVIN ADDED
std::vector<std::vector<gtid_t>>& graph::get_graph(){
  return m_graph;
}

// ALVIN ADDED
std::vector<gate>& graph::get_gate_list(){
  return m_gate_list;
}

void graph::insert_gate(gate const & new_gate)
{
    std::vector<gtid_t> adj_vertices{};
    m_graph.push_back(adj_vertices);

    // map the new gate to its index
    // cout << "Inserting " << new_gate.get_name() << endl;
    m_gate_list.push_back(new_gate);
}

void graph::insert_edge(gtid_t src, gtid_t dest)
{
    if (src > m_graph.size())
    {
        perror("gate does not exist.");
        exit(1);
    }
    m_graph[src].push_back(dest);
}

void graph::set_heights(std::vector<gtid_t> const & start_vertices)
{

}

void graph::print()
{
    const char separator = '\t';
    const uint8_t width_id = 3;
    const uint8_t width_name = 10;
    const uint8_t width_type = 6;
    cout << "ID\t(Name)\t\t(Type)\t\tAdjacent Gates(ID)" << endl;
    for (uint32_t i = 0; i < m_graph.size(); ++i)
    {
        gate cur_gate = m_gate_list[i];
        vector<gtid_t> adj_gates = m_graph[i];
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

        // ALVIN ADDED
        cout << " " <<  cur_gate.get_gate_level() << "" << cur_gate.get_gate_pos();
        
        cout << endl;
    }
}



}



