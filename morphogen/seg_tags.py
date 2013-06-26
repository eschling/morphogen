import io
import argparse
from collections import namedtuple

Analysis = namedtuple('Analysis', 'stem, tag')

def main():
    parser = argparse.ArgumentParser(description='Convert segmentations to tags')
    parser.add_argument('analyses', help='unsupervised segmentations')
    args = parser.parse_args()

    with io.open(args.analyses, encoding='utf8') as f:
        analyses = {}
        for line in f:
            word, analysis = line[:-1].split('\t')
            prefixes, rest = analysis.split('<')
            stem, suffixes = rest.split('>')
            if len(stem) > 3:
                prefix = '+'.join(p for p in prefixes.split('^') if p)
                if prefix: prefix = prefix+'+'
                suffix = '+'.join(s for s in suffixes.split('^') if s)
                if suffix: suffix = '+'+suffix
                tag = prefix+'STEM'+suffix
                analyses[word] = Analysis(stem, tag)

    out = io.open('/dev/stdout', 'w', encoding='utf8')
    for line in io.open('/dev/stdin', encoding='utf8'):
        words = line[:-1].lower().split()
        w = ' '.join(words)
        l = ' '.join(analyses[word].stem if word in analyses else word for word in words)
        a = ' '.join('W'+analyses[word].tag if word in analyses else 'X' for word in words)
        out.write(u'{} ||| {} ||| {}\n'.format(w, l, a))

if __name__ == '__main__':
    main()
