import logging
import re
from collections import namedtuple

AnnotatedToken = namedtuple('AnnotatedToken', 'token, pos, parent, dependency, cluster')

def extract_instances(category, source, target, alignment):
    """Extract (category, features, tag) training instances for a sentence pair"""
    for j, (token, lemma, tag) in enumerate(target):
        if tag[0] != category: continue
        word_alignments = [i for (i, k) in alignment if k == j] # src - tgt == j
        if len(word_alignments) != 1: continue # Extract only 1-n alignments
        (i,) = word_alignments # src
        features = dict((fname, fval) for ff in config.FEATURES
                for fname, fval in ff(source, lemma, i))
        yield (token, lemma, tag), features


def read_sentences(stream, skip_empty=True):
    """Read annotated sentences in the format:
    EN ||| EN POS ||| EN dep ||| EN clus ||| RU ||| RU lemma ||| RU tag ||| EN-RU alignment"""
    for line in stream:
        fields = line.decode('utf8')[:-1].split(' ||| ')
        src, src_pos, src_dep, src_clus, tgt, tgt_lem, tgt_tag, als = fields

        # Read target
        tgt = tgt.lower().split()
        tgt_lem = tgt_lem.lower().split()
        tgt_tag = tgt_tag.split()
        # check
        if skip_empty and not tgt_lem:
            logging.debug('Skip empty analysis for sentence: %s', ' '.join(tgt))
            continue
        if not len(tgt) == len(tgt_lem) == len(tgt_tag):
            logging.error('Bad annotation for "%s"', ' '.join(tgt))
            continue
        tgt_tokens = zip(tgt, tgt_lem, tgt_tag)

        # Read source
        src = src.lower().split()
        src_pos = src_pos.split()
        src_parents, src_dtypes = zip(*[(int(parent), typ) for parent, typ in 
            (dep.split('-') for dep in src_dep.split())])
        src_clus = src_clus.split()
        src_tokens = [AnnotatedToken(*info) for info in
                zip(src, src_pos, src_parents, src_dtypes, src_clus)]

        if not len(src) == len(src_pos) == len(src_parents) == len(src_dtypes) == len(src_clus):
            logging.error('Bad tag/parse for "%s" (%d/%d/%d/%d/%d)', ' '.join(src),
                    len(src), len(src_pos), len(src_parents), len(src_dtypes), len(src_clus))
            continue

        # Read alignment (src - tgt) [en - ru]
        alignments = [(int(i), int(j)) for i, j in
            (point.split('-') for point in als.split())]

        yield src_tokens, tgt_tokens, alignments

sentence_re = re.compile('^<seg grammar="([^"]+)" id="(\d+)">(.+)</seg>$')
fields_re = re.compile('\s*\|\|\|\s*')

def read_sgm(fn):
    with open(fn) as f:
        for line in f:
            fields = fields_re.split(line.decode('utf8')[:-1])
            path, sid, src = sentence_re.match(fields[0]).groups()
            yield path, sid, src, fields[1:]
