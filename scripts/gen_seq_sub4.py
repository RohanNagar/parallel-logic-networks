import argparse

from itertools import product


def main(n, outfile):
    final = 2**n
    with open(outfile, 'w') as f:
        f.write('set sim_time [time {run -all}]\n')
        for i in range(0,final):
            num = format(i, '0' + str(n) + 'b')
            f.write('force A 4\'b' + num[1:5]+'\n');
            f.write('force B 4\'b' + num[5:9]+'\n');
            f.write('force Bin 1\'b' + num[0:1]+'\n'); 
            f.write('run 1\n')  
      


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser('Binary String Generator')
    parser.add_argument('-n', '--length', type=int, required=True,
            help='the length of the binary strings to generate')
    parser.add_argument('-f', '--filename', type=str, required=True,
            help='the name of the file to create')

    args = parser.parse_args()
    main(args.length, args.filename)

