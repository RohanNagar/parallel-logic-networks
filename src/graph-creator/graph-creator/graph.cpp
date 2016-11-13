
#include "graph.h"

/**
 * Create an empty ordered map.
 */
pln::graph::graph() {
    m_graph = std::vector<std::vector<vertex_t>>{};
}

void pln::graph::insert_vertex(vertex_t id) {
    std::vector<vertex_t> adj_vertices;
    m_graph.push_back(adj_vertices);
}

void pln::graph::insert_edge(vertex_t src, vertex_t dest) {
    if (src > m_graph.size()) {
        perror("Vertex does not exist.");
        exit(1);
    }
    m_graph[src].push_back(dest);
}

void pln::graph::set_heights(std::vector<vertex_t> const& start_vertices) {

}
