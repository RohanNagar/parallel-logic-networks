
#pragma once

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_set>

#include "gate-type.h"

using gtid_t = int32_t;              // id type of a gate - just an int

namespace pln
{


/**
 * Gate class.
 */
class gate
{

private:
    gtid_t m_id;                         // gate id (just a uint32_t)
    std::string const m_name;           // unique name of the gate
    gate_type m_type;                   // gate type (enum)
    std::string const m_type_name;      // gate type (string)
    uint32_t m_gate_level;                // ALVIN ADDED
    uint32_t m_gate_pos;                  // ALVIN ADDED 

    static uint32_t m_num_gates;                            // count of the total number of gates
    static std::unordered_set<std::string> m_gate_lib;      // list of all the gate types

public:
    gate();
    gate(std::string const & name);
    gate(std::string const & name, std::string const & type);

    gtid_t get_id() const;
    std::string const & get_name() const;
    std::string const & get_type() const;
    gate_type get_gate_type();
    void set_gate_level(uint32_t); // ALVIN ADDED
    void set_gate_pos(uint32_t);   // ALVIN ADDED
    uint32_t get_gate_level(); // ALVIN ADDED
    uint32_t get_gate_pos();   // ALVIN ADDED

    static bool in_gate_lib(std::string const & type);
};

}
