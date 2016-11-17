
#pragma once

#include <vector>
#include <unordered_map>

#include "gate.h"

namespace pln
{


class module
{
private:
    std::string const m_name;

    std::vector<gid_t> m_input_ports;
    std::vector<gid_t> m_output_ports;
    std::unordered_map<std::string, gid_t> m_gate_list;             // maintain a gate list for each module that is indexed by each gate's name
    std::unordered_map<std::string, std::string> m_module_list;     // maintain an internal module list for each module


public:
    module(std::string const & name);

    std::string const & get_name() const;

    void insert_input_port(gate const & port);
    void insert_output_port(gate const & port);
    void insert_internal_module(std::string const & name, std::string const & mod_name);
    void insert_gate(gate const & gt);

    gid_t find_gate(std::string const & name);
    std::string const & find_internal_module(std::string const & name);

    static module module_err;                                   // module to return when not found
};




}

