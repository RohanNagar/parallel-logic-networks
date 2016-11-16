
#include "gate.h"

using std::string;
using std::vector;
using std::unordered_set;

namespace pln
{

uint32_t gate::m_num_gates = 0;         // initialize the gate count to 0
unordered_set<string> gate::m_gate_lib
{
    "NO_GATE",
    "PORT_I",
    "PORT_O",
    "OBUF",
    "INV",
    "AND",
    "ADD",
    "OR0",
    "XOR0",
    "NAND",
    "NOR",
    "NUM_GATES"
};


gate::gate() :
    gate("", "")
{
  
}

gate::gate(string const & name) :
    gate(name, "")
{

}


gate::gate(string const & name, string const & type) :
    m_id{ (int)gate::m_num_gates },
    m_name{ name },
    m_type_name{ type },
    m_gate_level{ 0 }, // ALVIN ADDED
    m_gate_pos{ 0 }   // ALVIND ADDED
{
    ++gate::m_num_gates;            // increment total gate count

    if (type == "NO_GATE")
    {
        m_type = gate_type::NO_GATE;
    }

    else if (type == "PORT_I")
    {
        m_type = gate_type::PORT_I;
    }

    else if (type == "PORT_O")
    {
        m_type = gate_type::PORT_O;
    }

    else if (type == "OBUF")
    {
        m_type = gate_type::OBUF;
    }

    else if (type == "INV")
    {
        m_type = gate_type::INV;
    }

    else if (type == "AND")
    {
        m_type = gate_type::AND;
    }

    else if (type == "ADD")
    {
        m_type = gate_type::ADD;
    }

    else if (type == "OR")
    {
        m_type = gate_type::OR;
    }

    else if (type == "XOR")
    {
        m_type = gate_type::XOR;
    }

    else if (type == "NAND")
    {
        m_type = gate_type::NAND;
    }

    else if (type == "NOR")
    {
        m_type = gate_type::NOR;
    }

    else
    {
        m_type = gate_type::NO_GATE;
    }

    
}




/**
 * Public member functions.
 */


gtid_t gate::get_id() const
{
    return m_id;
}

string const & gate::get_name() const
{
    return m_name;
}

string const & gate::get_type() const
{
    return m_type_name;
}

// ALVIN ADDED
void gate::set_gate_level(uint32_t level){
  m_gate_level = level;
}

void gate::set_gate_pos(uint32_t pos){
  m_gate_pos = pos;
}

uint32_t gate::get_gate_level(){
  return m_gate_level;
}

uint32_t gate::get_gate_pos(){
  return m_gate_pos;
}

bool gate::in_gate_lib(string const & type)
{
    auto it = gate::m_gate_lib.find(type);
    return it != gate::m_gate_lib.end();
}



}



