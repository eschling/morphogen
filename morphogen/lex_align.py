import sys
from common import read_sentences

def main():
    for source, target, alignment in read_sentences(sys.stdin):
        src = ' '.join(w.token for w in source)
        tgt = ' '.join(lemma+'_'+tag[0] for _, lemma, tag in target)
        print(u'{} ||| {}'.format(src, tgt).encode('utf8'))

if __name__ == '__main__':
    main()

