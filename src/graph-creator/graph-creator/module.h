
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
    module(std::string const name);
};




}

