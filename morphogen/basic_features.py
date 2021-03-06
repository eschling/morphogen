WINDOW_SIZE = 1
def window_words(source, target_lemma, alignment):
    """source window of words around alignment point"""
    for src in range(max(alignment-WINDOW_SIZE, 0), min(alignment+WINDOW_SIZE+1, len(source))):
        if src == alignment: continue
        prefix = 'before' if src < alignment else 'after'
        yield prefix+'_'+source[src].token, 1
        yield prefix+'_pos_'+source[src].pos, 1
        yield prefix+'_cluster_'+source[src].cluster, 1

def basic_dependency(source, target_lemma, alignment):
    yield 'src_'+source[alignment].token.lower(), 1
    yield 'src_pos_'+source[alignment].pos, 1
    yield 'src_cluster_'+source[alignment].cluster, 1
    deptype = source[alignment].dependency
    if deptype == 'ROOT':
        yield 'src_root', 1
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

def advanced_dependency(source, target_lemma, alignment):
    yield 'token_'+source[alignment].token, 1
    yield 'pos_'+source[alignment].pos, 1
    yield 'cluster_'+source[alignment].cluster, 1

    deptype = source[alignment].dependency
    if deptype == 'ROOT':
        parent = -1
        yield 'is_root', 1
    else:
        parent = source[alignment].parent - 1
        yield 'deptype_'+deptype, 1
        yield 'parent_token_'+source[parent].token, 1
        yield 'parent_pos_'+source[parent].pos, 1
        yield 'parent_cluster_'+source[parent].cluster, 1
        parent_deptype = source[parent].dependency
        if parent_deptype == 'ROOT':
            yield 'src_parent_is_root', 1
        else:
            grandparent = source[parent].parent - 1
            yield 'parent_deptype_'+parent_deptype, 1
            yield 'grandparent_token_'+source[grandparent].token, 1
            yield 'grandparent_pos_'+source[grandparent].pos, 1
            yield 'grandparent_cluster_'+source[grandparent].cluster, 1
            
    n_child = 0
    n_siblings = 0
    for i, src in enumerate(source):
        if src.parent - 1 == alignment: # found a child
            n_child += 1
            yield 'child_'+src.dependency, 1
            yield 'child_'+src.dependency+'_token_'+src.token, 1
            yield 'child_'+src.dependency+'_pos_'+src.pos, 1
            yield 'child_'+src.dependency+'_cluster_'+src.cluster, 1
        if src.parent - 1 == parent and i != alignment: # found a sibling
            n_siblings += 1
    yield 'n_child', n_child
    yield 'n_siblings', n_siblings
