import basic_features

# List of feature functions to use for extraction
FEATURES = [basic_features.advanced_dependency, basic_features.window_words]

def get_attributes(cat, attrs):
    prefix, suffix = [], []
    position = 0
    for part in attrs.split('+'):
        if part == 'STEM':
            position += 1
        else:
            (prefix, suffix)[position].append(part)
    for i, v in enumerate(reversed(prefix), 1):
        yield u'Prefix_{}_{}'.format(i, v)
    for i, v in enumerate(suffix, 1):
        yield u'Suffix_{}_{}'.format(i, v)
