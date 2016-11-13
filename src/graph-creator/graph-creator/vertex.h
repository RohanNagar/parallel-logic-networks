
#pragma once

#include <cstdlib>
#include <cstdint>
#include <map>

using vertex_t = uint32_t;          // id type of a vertex - just an unsigned int
using bit_t = uint8_t;              // single logic bit - 0 or 1


namespace pln {

class graph;

/**
 * Vertex class.
 */
class vertex {

private:
    static uint32_t m_num_vertices;

    vertex_t m_id;

public:
    vertex();
    bit_t eval();

        
    vertex_t get_id();
};

}
