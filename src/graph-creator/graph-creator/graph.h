
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
    std::vector<std::vector<gtid_t>> m_graph;        // adjacencly list structure
    std::vector<gate> m_gate_list;                  // list of all gates; index is the gate id
    std::vector<module> m_module_list;              // list of all modules; the last module will be the overall module
    uint32_t m_max_level;
    uint32_t m_max_width;

public:
    graph();

    void insert_gate(gate const & new_gate);
    void insert_edge(gtid_t src, gtid_t dest);
    void insert_module(module const & mod);

    std::vector<std::vector<gtid_t>>& get_graph();  // ALVIN ADDED
    std::vector<gate>& get_gate_list();             // ALVIN ADDED

    module const & find_module(std::string const & name);

    void set_heights(std::vector<gtid_t> const & start_vertices);

    void print();

};





}

