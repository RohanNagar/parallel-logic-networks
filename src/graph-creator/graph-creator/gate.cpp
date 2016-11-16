
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
    "OR",
    "XOR",
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
    m_id{ gate::m_num_gates }
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


gid_t gate::get_id()
{
    return m_id;
}



bool gate::in_gate_lib(string const & type)
{
    auto it = gate::m_gate_lib.find(type);
    return it != gate::m_gate_lib.end();
}



}



