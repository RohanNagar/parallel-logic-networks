import argparse

class Cell():

    def __init__(self, name):
        self.name = name
        self.inputs = list()
        self.outputs = list()
        self.instances = list()
        self.nets = list()

    def add_input(self, name):
        self.inputs.append(name)

    def add_output(self, name):
        self.outputs.append(name)

    def add_instance(self, name):
        self.instances.append(name)

    def add_net(self, name):
        self.nets.append(name)

    def __str__(self):
        return (self.name
                  + ' -i ' + ' '.join(name for name in self.inputs)
                  + ' -o ' + ' '.join(name for name in self.outputs)
                )

def main(infile_name, outfile_name):
    # Open the file to read
    with open(infile_name, 'r') as infile, open(outfile_name, 'w') as outfile:
        current_cell = None

        for line in infile:
            words = line.strip().split(' ')
            words = [word.strip('(').strip(')') for word in words]

            # If we have a new cell, write the current one to the file
            # and start creating the new one
            if words[0] == 'cell':
                if current_cell is not None:
                    outfile.write(str(current_cell) + '\n')

                current_cell = Cell(words[1])

            # Check for items to add to the cell
            if words[0] == 'port':
                if words[3] == 'OUTPUT':
                    current_cell.add_output(words[1])
                elif words[3] == 'INPUT':
                    current_cell.add_input(words[1])
                else:
                    print('Um... something went wrong')

        # Write the last cell
        outfile.write(str(current_cell))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EDN File Parser')

    # Add command line arguments
    parser.add_argument('-f', '--filename', type=str, required=True, help='the name of the file to parse')
    parser.add_argument('-o', '--outfile', type=str, required=True, help='the name of the file to create')
    args = parser.parse_args()

    main(args.filename, args.outfile)

