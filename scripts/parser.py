import argparse


class Net():
    ''' Defines a network connection.
    
        Each network connection is composed of
            - an output node
            - a set of input nodes
    '''
    
    def __init__(self, name):
        self.name = name
        self.output = None
        self.inputs = list()

    def set_output(self, name):
        self.output = name

    def add_input(self, name):
        self.inputs.append(name)

    def swap(self):
        ''' Swaps the output and input.
        
            Should only be used when there is a single input. If there 
            is more than one input, the first input in the list is 
            swapped with the output.
        '''
        
        self.inputs.append(self.output)
        self.output = self.inputs[0]
        del self.inputs[0]

    def check_output(self, current_cell, all_cells):
        ''' Checks if self.output is not None.
            If it is none, this method then attempts to
            find an output port from the list of inputs.
        '''
        
        # If we already have an output, we're fine
        if self.output is not None:
            return True

        index = None
        for idx, inp in enumerate(self.inputs):
            name, port = inp.split(' ')
            typ = None

            # Find the type of the instance that matches the input name
            instances = current_cell.get_instances()
            for instance in instances:
                if instance.split(' ')[0] == name:
                    typ = instance.split(' ')[1]

            if typ is None:
                print('No appropriate type found.')
                continue

            # Go through all known types and see if the input port
            # corresponds to one of that type's output ports
            for cell in all_cells:
                if cell.get_name() == typ and cell.contains_output(port):
                    self.output = inp
                    index = idx
                    break

            # If we have set the index value, then we're done
            if index is not None:
                break

        # Return false if no ports were found
        if index is None:
            return False

        # Remove the set output from the inputs list
        del self.inputs[index]
        return True

    def __str__(self):
        result = self.output.strip(' O ') + ' <-'

        for inp in self.inputs:
            result += ' ' + inp + ','

        return result.rstrip(',')


class Cell():

    def __init__(self, name):
        self.name = name
        self.inputs = list()
        self.outputs = list()
        self.instances = list()
        self.nets = list()

    def get_name(self):
        return self.name

    def add_input(self, name):
        self.inputs.append(name)

    def add_output(self, name):
        self.outputs.append(name)

    def add_instance(self, name):
        self.instances.append(name)

    def get_instances(self):
        return self.instances

    def add_net(self, name):
        self.nets.append(name)

    def contains_output(self, name):
        return (name in self.outputs)

    def __str__(self):
        result = (self.name
                  + ' -i ' + ' '.join(name for name in self.inputs)
                  + ' -o ' + ' '.join(name for name in self.outputs))

        if self.instances:
            result += '\n\tInstances:'
            for instance in self.instances:
                result += ('\n\t\t' + instance)

        if self.nets:
            result += '\n\tNets:'
            for net in self.nets:
                result += ('\n\t\t' + str(net))

        return result


def add_array_ports(cell, words, sizes):
    name = words[3]
    typ = words[len(words) - 1]
    start, stop = words[4].split(':')
    start, stop = int(start[len(start) - 1]), int(stop[0]) - 1

    sizes[name] = start

    for i in range(start, stop, -1):
        if typ == 'INPUT':
            cell.add_input(name + str(i))
        elif typ == 'OUTPUT':
            cell.add_output(name + str(i))
        else:
            print('Error adding array ports to cell.')


def add_net_to_cell(net, cell, all_cells):
    if net is None:
        return

    if net.check_output(cell, all_cells):
        cell.add_net(net)
    else:
        print('Could not find an output for the current net')


def main(infile_name, outfile_name):
    # Open the file to read
    with open(infile_name, 'r') as infile, open(outfile_name, 'w') as outfile:
        current_cell = None
        current_net = None

        all_cells = list()
        arr_sizes = dict()

        for line in infile:
            words = line.strip().split(' ')
            words = [word.strip('(').strip(')') for word in words]

            # If we have a new cell, write the current one to the file
            # and start creating the new one
            if words[0] == 'cell':
                if current_cell is not None:
                    add_net_to_cell(current_net, current_cell, all_cells)
                    all_cells.append(current_cell)
                    outfile.write(str(current_cell) + '\n\n')

                current_cell = Cell(words[1])

            # Check for items to add to the cell
            if words[0] == 'port':
                last = len(words) - 1

                if words[1] == 'array':
                    add_array_ports(current_cell, words, arr_sizes)
                else:
                    if words[last] == 'OUTPUT':
                        current_cell.add_output(words[1])
                    elif words[last] == 'INPUT':
                        current_cell.add_input(words[1])
                    else:
                        print('Um... something went wrong')

            if words[0] == 'instance':
                # Renamed instances have different positions
                if words[1] == 'rename':
                    current_cell.add_instance(words[2] + ' ' + words[7])
                else:
                    current_cell.add_instance(words[1] + ' ' +  words[5])

            if words[0] == 'net':
                add_net_to_cell(current_net, current_cell, all_cells)
                
                # Create a new net
                if words[1] == 'rename':
                    current_net = Net(words[2])
                else:
                    current_net = Net(words[1])

            if words[0] == 'portref':
                if words[1] == 'member':
                    arr_name = words[2]
                    idx = int(words[3])
                    name = arr_name + str(arr_sizes[arr_name] - idx)
                    
                    current_net.set_output(name)
                    if current_cell.contains_output(name):
                        current_net.swap()
                elif len(words) > 2:
                    current_net.add_input(words[3] + ' ' + words[1])
                else:
                    current_net.set_output(words[1])
                    if current_cell.contains_output(words[1]):
                        current_net.swap() 

        # Write the last cell
        add_net_to_cell(current_net, current_cell, all_cells)
        outfile.write(str(current_cell))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EDN File Parser')

    # Add command line arguments
    parser.add_argument('-f', '--filename', type=str, required=True, help='the name of the file to parse')
    parser.add_argument('-o', '--outfile', type=str, required=True, help='the name of the file to create')
    args = parser.parse_args()

    main(args.filename, args.outfile)

