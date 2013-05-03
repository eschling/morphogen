from itertools import groupby

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

tag_length = {category: sum(1 for _ in group) for category, group in groupby(sorted(attributes), key=lambda t:t[0])}
