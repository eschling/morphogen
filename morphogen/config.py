#import abstract_features
import basic_features

# List of features function to use for extraction
#FEATURES = [abstract_features.abstract_dependency, abstract_features.abstract_window_words]
FEATURES = [basic_features.basic_dependency, basic_features.window_words]
# List of POS categories to extract training data for
EXTRACTED_TAGS = 'NVARM'

import tagset
def get_attributes(cat, attrs):
    category = tagset.categories[cat]
    for i, attr in enumerate(attrs, 1):
        if attr != '-':
            yield tagset.attributes[category, i]+'_'+attr
