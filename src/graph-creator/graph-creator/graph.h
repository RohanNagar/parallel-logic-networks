
#pragma once

#include <map>
#include <vector>
#include <unordered_set>

#include "gate.h"
#include "module.h"

namespace pln
{


/**
 * We represent the combinational circuit using graph using an adjacency list. The adjacency list looks like:
 *
 *  gate      List of Adjacent vertices (Vector)
 *  0           1, 2
 *  1           4, 3
 *  2           3
 *  3           4
 *  4           <empty>
 *
 * To represent the whole structure above, it is easy to use a containing vector from the stl.
 */
class graph
{

private:
    std::vector<std::vector<gid_t>> m_graph;
    std::vector<gate> m_gate_list;

    std::vector<module> m_module_list;

    std::vector<std::string> m_gate_types;

public:
    graph();

    void insert_gate(gate const & new_gate);

    void insert_edge(gid_t src, gid_t dest);

    void set_heights(std::vector<gid_t> const & start_vertices);

};





}

