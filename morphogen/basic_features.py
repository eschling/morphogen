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

    parent = alignment
    gen = 0 # parent is always gen generations above the current node

    deptype = source[parent].dependency
    prev_parents = set()
    while deptype != 'ROOT':
        gen += 1
        if gen > 1:
            break # safeguard against insane dep trees
        prev_parents.add(parent)

        parent = source[parent].parent - 1
        if parent in prev_parents:
           break # in case of cycles in dep tree
        yield 'src_%d_deptype_%s' %(gen, deptype), 1
        yield 'src_%d_parent_%s' % (gen, source[parent].token), 1
        yield 'src_%d_parent_pos_%s' % (gen, source[parent].pos), 1
        yield 'src_%d_parent_cluster_%s' % (gen, source[parent].cluster), 1

    if deptype == 'ROOT':
       yield 'src_%d_parent_root' % gen, 1

    n_child = 0
    for src in source:
        if src.parent - 1 == alignment:
            n_child += 1
            yield 'src_child_'+src.dependency, 1
            yield 'src_child_'+src.dependency+'_'+src.token, 1
            yield 'src_child_'+src.dependency+'_'+src.pos, 1
            yield 'src_child_'+src.dependency+'_'+src.cluster, 1
    yield 'src_n_child', n_child
