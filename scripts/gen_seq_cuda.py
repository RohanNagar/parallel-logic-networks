import argparse

from itertools import product


def main(n, outfile):
    final = 2**n
    with open(outfile, 'w') as f:
        f.write(str(final) + '\n');  
        for i in range(0,final):
            f.write(format(i, '0' + str(n) +  'b') + '\n')    
      


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser('Binary String Generator')
    parser.add_argument('-n', '--length', type=int, required=True,
            help='the length of the binary strings to generate')
    parser.add_argument('-f', '--filename', type=str, required=True,
            help='the name of the file to create')

    args = parser.parse_args()
    main(args.length, args.filename)

