'''Example configuration file, for use with a positional tagset for Russian.
For example, a Russian word might be tagged as mis-sfm-e 
(the morphosyntactic description of the features, which is equivalent to 
main+indicative+past+singular+feminine+medial+perfective)

All configuration files must contain get_attributes, which specifies how to 
name features based on the target language preprocessed morphological analysis. 
This function takes a category and a tag 
(the output of the morphological analyzer for a given word) and yields a list of features.
The features returned by the function are used to define the target morphological feature vectors. 
Configuration files must be present in this configuration directory.

You can train a model with this configuration by running 
(assuming corpus is a training set in the format: SRC ||| SRC tag ||| SRC deps ||| TGT ||| TGT lemma ||| TGT tag ||| alignments, 
 and rev_map is the output from rev_map.py)
 
by running (for verbs, for example)
cat corpus | python struct_train.py -r 0.01 -i 10 -c russian_config V rev_map model
'''
from itertools import groupby

#EXTRACTED_TAGS = 'NVARM'

def get_attributes(cat, attrs):
  category = categories[cat]
  for i, attr in enumerate(attrs, 1):
    if attr != '-':
      yield attributes[category, i]+'_'+attr

_categories = """
Noun N
Verb V
Adjective A
Adverb R
Numeral M
"""

categories = {cat: category for category, cat in
        (line.split() for line in _categories.strip().split('\n'))}

_attributes = """
Degree  Adverb  1
Type    Numeral 1
Gender  Numeral 2
Number  Numeral 3
Case    Numeral 4
Form    Numeral 5
Type    Adjective   1
Degree  Adjective   2
Gender  Adjective   3
Number  Adjective   4
Case    Adjective   5
Definiteness    Adjective   6
Type    Noun    1
Gender  Noun    2
Number  Noun    3
Case    Noun    4
Animate Noun    5
Case2   Noun    6
Type    Verb    1
VForm   Verb    2
Tense   Verb    3
Person  Verb    4 
Number  Verb    5
Gender  Verb    6
Voice   Verb    7
Definiteness    Verb    8
Aspect  Verb    9
Case    Verb    10 
"""

attributes = {(category, int(pos)): attribute for attribute, category, pos in
        (line.split() for line in _attributes.strip().split('\n'))}
