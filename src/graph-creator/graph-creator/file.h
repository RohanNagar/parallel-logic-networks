
#pragma once

#include <string>

#include "graph.h"
#include "gate.h"

namespace pln
{


class file_parser
{

private:
    std::string m_filename_in;
    std::string m_filename_out;

public:
    file_parser(std::string const & filename_in, std::string const & filename_out);

    void parse(graph & g);
};


}

