
#pragma once

#include <map>
#include <vector>

#include "vertex.h"

namespace pln {


/**
 * We represent the graph using an adjacency list. The adjacency list looks like:
 * 
 *  Vertex        List of Adjacent vertexs (Vector)
 *  0           1, 2, 3
 *  1           4, 3
 *  2           3
 *  3           4
 *  4           <empty>
 *
 * To represent the whole structure above, it is easy to use the ordered map from stl.
 */
class graph {

private:
    std::map<vertex_t, std::vector<vertex_t>> m_graph;

public:
    graph();

    void insert_vertex(vertex_t id);
    void insert_edge(vertex_t src, vertex_t dest);

    void set_heights(std::vector<vertex_t> const& start_vertices);

};





}

