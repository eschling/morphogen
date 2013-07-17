#!/usr/bin/env python
import sys
import os
import subprocess as sp

def main(turboparse):
    os.environ['LD_LIBRARY_PATH'] = turboparse+'/deps/local/lib'
    turbotag = turboparse+'/src/tagger/TurboTagger'
    sys.stderr.write(turbotag+'\n') 
    model = turboparse+'/models/english_proj_tagger.model'
    tagger = sp.Popen([turbotag, '--test', '--file_model='+model,
        '--file_test=/dev/stdin', '--file_prediction=/dev/stdout'],
        stdin=sp.PIPE, stdout=sp.PIPE)

    def tag(words):
        tagger.stdin.write('\n'.join('{}\t_'.format(word) for word in words)+'\n\n')
        tagger.stdin.flush()
        tags = []
        line = tagger.stdout.readline()
        while line != '\n':
            tags.append(line.split()[1])
            line = tagger.stdout.readline()
        assert len(words) == len(tags)
        return ' '.join(tags)

    for line in sys.stdin:
        sentence = line[:-1]
        print('{} ||| {}'.format(sentence, tag(sentence.split())))

if __name__ == '__main__':
    main(sys.argv[1])

