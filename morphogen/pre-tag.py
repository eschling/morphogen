import sys
import logging
import analyzer
import multiprocessing

def sentences(stream):
    for line in stream:
        yield line.decode('utf8')[:-1].split(' ||| ')

tagger = None
def make_tagger():
    global tagger
    tagger = analyzer.Tagger()

def tag_sentence(x):
    global tagger
    source, target, alignment = x
    try:
        return x, list(tagger.tag(target.split()))
    except analyzer.AnalysisError as e:
        logging.error('Tagging failed for "%s": %s', target, e)
        return None

def main():

    pool = multiprocessing.Pool(processes=20, initializer=make_tagger)

    for y in pool.imap(tag_sentence, sentences(sys.stdin)):
        if not y: continue
        (source, target, alignment), analyses = y

        tags = ' '.join(analysis.tag for analysis in analyses)
        lemmas = ' '.join(analysis.lemma for analysis in analyses)

        out = u'{} ||| {} ||| {} ||| {} ||| {}'.format(source, target, lemmas, tags, alignment)
        print(out.encode('utf8'))

if __name__ == '__main__':
    main()

