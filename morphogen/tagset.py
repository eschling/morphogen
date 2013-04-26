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
Animate Noun    5
Animate Numeral 9
Aspect  Verb    9
Case    Adjective   5
Case    Noun    4
Case    Numeral 4
Case    Verb    10
Case2   Noun    6
Definiteness    Adjective   6
Definiteness    Verb    8
Degree  Adjective   2
Degree  Adverb  1
Form    Numeral 5
Gender  Adjective   3
Gender  Noun    2
Gender  Numeral 2
Gender  Verb    6
Number  Adjective   4
Number  Noun    3
Number  Numeral 3
Number  Verb    5
Person  Verb    4
Tense   Verb    3
Type    Adjective   1
Type    Noun    1
Type    Numeral 1
Type    Verb    1
VForm   Verb    2
Voice   Verb    7
"""

attributes = {(category, int(pos)): attribute for attribute, category, pos in 
        (line.split() for line in _attributes.strip().split('\n'))}
