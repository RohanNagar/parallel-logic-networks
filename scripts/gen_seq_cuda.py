import argparse

from itertools import product


def main(n, outfile):
    sequences = [''.join(seq) for seq in product('01', repeat=n)]

    with open(outfile, 'w') as f:
        f.write(str(2**n) + '\n')
        f.write('\n'.join(seq for seq in sequences))


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser('Binary String Generator')
    parser.add_argument('-n', '--length', type=int, required=True,
            help='the length of the binary strings to generate')
    parser.add_argument('-f', '--filename', type=str, required=True,
            help='the name of the file to create')

    args = parser.parse_args()
    main(args.length, args.filename)

