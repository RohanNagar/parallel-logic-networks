
#include "module.h"

using std::string;
using std::vector;

namespace pln
{


module::module(string const & name) :
    m_name{ name },
    m_input_ports{ vector<gid_t>{} },
    m_output_ports{ vector<gid_t>{} }
{

}

void module::add_input_port(gid_t port)
{
    m_input_ports.push_back(port);
}


void module::add_output_port(gid_t port)
{
    m_output_ports.push_back(port);
}


}

