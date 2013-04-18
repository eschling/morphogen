def words(source, j, analysis, alignment):
    for word in set(source):
        yield word, 1
