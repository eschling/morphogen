def bow(source, target_lemma, alignment):
    """source bag-of-words feature"""
    for token in set(word.token for word in source):
        yield token, 1

WINDOW_SIZE = 2
def window_words(source, target_lemma, alignment):
    """source window of words around alignment point"""
    for src in range(max(alignment-WINDOW_SIZE, 0), min(alignment+WINDOW_SIZE+1, len(source))):
        yield source[src].token, 1
