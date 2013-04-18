#coding:utf8
import subprocess as sp
from itertools import izip
from collections import namedtuple
import pymorphy2

TAGGER = '/home/vchahune/tools/tree-tagger/bin/tree-tagger'
PARAMS = '/home/vchahune/tools/tree-tagger/russian.par'

Analysis = namedtuple('Analysis', 'word, lemma, tag')

class AnalysisError:
    def __init__(self, i, o):
        self.i = i
        self.o = o

    def __repr__(self):
        return 'Analysis error: input={} words output={} words'.format(self.i, self.o)

class Tagger:
    def __init__(self):
        self.guesser = pymorphy2.MorphAnalyzer()

    def guess(self, word):
        parses = self.guesser.parse(word)
        if not parses:
            return word
        else:
            return max(parses, key=lambda parse:parse.estimate).normal_form

    def tag(self, words):
        proc = sp.Popen([TAGGER, '-lemma', '-token', PARAMS], stdin=sp.PIPE,
                stdout=sp.PIPE, stderr=sp.PIPE)
        proc.stdin.write('\n'.join(words).encode('utf8'))
        proc.stdin.close()

        analyzes = proc.stdout.readlines()
        if len(words) != len(analyzes):
            raise AnalysisError(len(words), len(analyzes))

        for word, analysis in izip(words, analyzes):
            w, tag, lemma = analysis.decode('utf8').split()
            assert word == w
            if lemma == '<unknown>':
                lemma = self.guess(word)
            yield Analysis(word, lemma, tag)

if __name__ == '__main__':
    tagger = Tagger()
    s = u'при этом единственными документами , поттеровская вселенная - это царство смерти .'
    print ' '.join((a.lemma+'+'+a.tag) for a in tagger.tag(s.split())).encode('utf8')
