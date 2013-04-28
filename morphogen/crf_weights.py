import argparse
import gzip
import re

fid_re = re.compile('^F(\d+) 0$')
def main():
    parser = argparse.ArgumentParser(description='Create cdec grammars')
    parser.add_argument('info', help='file containing initial weight information')
    parser.add_argument('weights', help='cdec weights')
    args = parser.parse_args()

    names = {}
    with open(args.info) as f:
        for line in f:
            assert line.startswith('# ')
            fname = line.decode('utf8')[2:-1]
            line = next(f)
            fid = int(fid_re.match(line).group(1))
            names[fid] = fname

    with gzip.open(args.weights) as f:
        for line in f:
            if line.startswith('#'): continue
            fid, fval = line[:-1].split(' ')
            print(u'{} {}'.format(names[int(fid[1:])], fval).encode('utf8'))

if __name__ == '__main__':
    main()

