def bow(source, target_lemma, alignment):
    """source bag-of-words feature"""
    for token in set(word.token for word in source):
        yield token, 1

WINDOW_SIZE = 1
def window_words(source, target_lemma, alignment):
    """source window of words around alignment point"""
    for src in range(max(alignment-WINDOW_SIZE, 0), min(alignment+WINDOW_SIZE+1, len(source))):
        if src == alignment: continue
        prefix = 'before' if src < alignment else 'after'
        yield prefix+'_'+source[src].token.lower(), 1
        yield prefix+'_pos_'+source[src].pos, 1
        yield prefix+'_cluster_'+source[src].cluster, 1

# TODO parent is root, n siblings
# TODO position features (delta pos in parse tree / relative position in sentence)

def basic_dependency(source, target_lemma, alignment):
    yield 'src_'+source[alignment].token.lower(), 1
    yield 'src_pos_'+source[alignment].pos, 1
    yield 'src_cluster_'+source[alignment].cluster, 1
    deptype = source[alignment].dependency
    if deptype == 'ROOT':
        yield 'src_is_root', 1
    else:
        parent = source[alignment].parent - 1
        yield 'src_deptype_'+deptype, 1
        yield 'src_parent_'+source[parent].token.lower(), 1
        yield 'src_parent_pos_'+source[parent].pos, 1
        yield 'src_parent_cluster_'+source[parent].cluster, 1
    n_child = 0
    for src in source:
        if src.parent - 1 == alignment:
            n_child += 1
            yield 'src_child_'+src.dependency, 1
            yield 'src_child_'+src.dependency+'_'+src.token.lower(), 1
            yield 'src_child_'+src.dependency+'_'+src.pos, 1
            yield 'src_child_'+src.dependency+'_'+src.cluster, 1
    yield 'src_n_child', n_child
