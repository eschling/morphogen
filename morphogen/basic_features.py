def bow(source, target_lemma, alignment):
    """source bag-of-words feature"""
    for token in set(word.token for word in source):
        yield token, 1

WINDOW_SIZE = 2
def window_words(source, target_lemma, alignment):
    """source window of words around alignment point"""
    for src in range(max(alignment-WINDOW_SIZE, 0), min(alignment+WINDOW_SIZE+1, len(source))):
        yield source[src].token, 1

def basic_dependency(source, target_lemma, alignment):
    yield 'src_'+source[alignment].token, 1
    yield 'src_pos_'+source[alignment].pos, 1
    yield 'src_cluster_'+source[alignment].cluster, 1
    deptype = source[alignment].dependency
    if deptype == 'ROOT':
        yield 'src_root', 1
    else:
        parent = source[alignment].parent - 1
        yield 'src_deptype_'+deptype, 1
        yield 'src_parent_'+source[parent].token, 1
        yield 'src_parent_pos_'+source[parent].pos, 1
        yield 'src_parent_cluster_'+source[parent].cluster, 1
    n_child = 0
    for src in source:
        if src.parent - 1 == alignment:
            n_child += 1
            yield 'src_child_'+src.dependency, 1
            yield 'src_child_'+src.dependency+'_'+src.token, 1
            yield 'src_child_'+src.dependency+'_'+src.pos, 1
            yield 'src_child_'+src.dependency+'_'+src.cluster, 1
    yield 'src_n_child', n_child
