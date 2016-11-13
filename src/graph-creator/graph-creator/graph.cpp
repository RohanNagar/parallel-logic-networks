
#include "graph.h"

/**
 * Create an empty ordered map.
 */
pln::graph::graph() {
    m_graph = std::map<vertex_t, std::vector<vertex_t>>();
}

void pln::graph::insert_vertex(vertex_t id) {
    std::vector<vertex_t> adj_list;
    m_graph.insert({ id, adj_list });
}

void pln::graph::insert_edge(vertex_t src, vertex_t dest) {
    m_graph.find(src);
}

void pln::graph::set_heights(std::vector<vertex_t> const& start_vertices) {

}
