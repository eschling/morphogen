def bow(source, target_lemma, alignment):
    """source bag-of-words feature"""
    for word in set(source):
        yield word, 1

WINDOW_SIZE = 2
def window_words(source, target_lemma, alignment):
    """source window of words around alignment point"""
    for src in range(max(alignment-WINDOW_SIZE, 0), min(alignment+WINDOW_SIZE+1, len(source))):
        yield source[src], 1
