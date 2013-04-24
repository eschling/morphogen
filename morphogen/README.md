Preprocessed parallel data format:
EN ||| EN POS ||| EN dep ||| EN clus ||| RU ||| RU lemma ||| RU tag ||| alignment

Feature function interface:
```python
def my_features(source, target_lemma, alignment):
    yield 'feature1', 1
    yield 'feature2', 1
```

Where:

- `source` is a list of AnnotatedToken(token, pos, parent, dependency, cluster)
- target_lemma is the lemma of the focus target word
- alignment is the alignment point of the focus target word in the source
