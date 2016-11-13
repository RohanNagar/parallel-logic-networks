
#include "vertex.h"

uint32_t pln::vertex::m_num_vertices = 0;        // initialize the vertex count to 0

pln::vertex::vertex() {
    m_id = pln::vertex::m_num_vertices;
    ++pln::vertex::m_num_vertices;
}
