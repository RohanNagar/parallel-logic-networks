
#include "graph.h"

/**
 * Create an empty ordered map.
 */
pln::graph::graph() :
    m_graph         { std::vector<std::vector<gid_t>>{} },
    m_gate_list     { std::vector<gate>{} }
{

}

void pln::graph::insert_gate(gate const & new_gate)
{
    std::vector<gid_t> adj_vertices{};
    m_graph.push_back(adj_vertices);

    // map the new gate to its index
    m_gate_list.push_back(new_gate);
}

void pln::graph::insert_edge(gid_t src, gid_t dest)
{
    if (src > m_graph.size())
    {
        perror("gate does not exist.");
        exit(1);
    }
    m_graph[src].push_back(dest);
}

void pln::graph::set_heights(std::vector<gid_t> const & start_vertices)
{

}

