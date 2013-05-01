import logging
from collections import namedtuple

AnnotatedToken = namedtuple('AnnotatedToken', 'token, pos, parent, dependency, cluster')

def read_sentences(stream):
    """Read annotated sentences in the format:
    EN ||| EN POS ||| EN dep ||| EN clus ||| RU ||| RU lemma ||| RU tag ||| alignment"""
    for line in stream:
        fields = line.decode('utf8')[:-1].split(' ||| ')
        src, src_pos, src_dep, src_clus, tgt, tgt_lem, tgt_tag, als = fields

        # Read target
        tgt = tgt.split()
        tgt_lem = tgt_lem.split()
        tgt_tag = tgt_tag.split()
        # check
        if not tgt_lem:
            logging.debug('Skip empty analysis for sentence: %s', ' '.join(tgt))
            continue
        if not len(tgt) == len(tgt_lem) == len(tgt_tag):
            logging.error('Bad annotation for "%s"', ' '.join(tgt))
            continue
        tgt_tokens = zip(tgt, tgt_lem, tgt_tag)

        # Read source
        src = src.split()
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

        # Read alignment (tgt - src) [ru - en]
        alignments = [(int(i), int(j)) for i, j in
            (point.split('-') for point in als.split())]

        yield src_tokens, tgt_tokens, alignments

