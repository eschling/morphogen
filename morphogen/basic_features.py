# TODO only take the words within a window around aligned words
def words(source, j, analysis, alignment):
    for word in set(source):
        yield word, 1
