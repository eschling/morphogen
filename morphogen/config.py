import basic_features

# List of features function to use for extraction
FEATURES = [basic_features.advanced_dependency, basic_features.window_words]
# List of POS categories to extract training data for
EXTRACTED_TAGS = 'NVARM'

import tagset
def get_attributes(cat, attrs):
    category = tagset.categories[cat]
    for i, attr in enumerate(attrs, 1):
        if attr != '-':
            yield tagset.attributes[category, i]+'_'+attr

"""
import basic_features

# List of features function to use for extraction
FEATURES = [basic_features.advanced_dependency, basic_features.window_words]
# List of POS categories to extract training data for
EXTRACTED_TAGS = 'W'

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
"""
