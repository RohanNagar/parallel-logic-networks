
#pragma once

#include <vector>

#include "gate.h"

namespace pln
{


class module
{
private:
    std::string const m_name;

    std::vector<gid_t> m_input_ports;
    std::vector<gid_t> m_output_ports;

public:
    module(std::string const & name);

    void add_input_port(gid_t port);
    void add_output_port(gid_t port);
};




}

