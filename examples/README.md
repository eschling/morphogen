## Examples

All example workflows are set up to work with [cdec](www.cdec-decoder.org), [TurboParser](http://www.ark.cs.cmu.edu/TurboParser/) (and the included CRF Tagger TurboTagger). We use Brown Clusters (available [here](http://www.ark.cs.cmu.edu/cdyer/en-c600.gz)) to get richer source side contextual features. You should download the brown clusters and point the global variable in whichever .tape file you're using to their location. 

The .tape files all contain detailed comments which should be read before running. Some known issues with fast_umorph are mentioned in the dependencies section of the main README file. 

unsupervised.tape : run morphogen, using fast_umorph to get unsupervised morphological segmentations.


supervised.tape : run morphogen, providing your own morphological analysis of the target side data

intrinsic.tape : sample script for intrinsic evaluation of inflection models. This script is mostly just preprocessing the dev/test data that you're evaluating on in the same manner as the training data in the other workflows (stemming, etc.). It is aligned along with the training data to produce more reliable alignments. This can all be added into the above workflows if you would rather perform the intrinsic evaluations automatically.

## Data

The data folder contains our Russian dev and test sets, and a small portion of our supervised russian training set (3000 sentences), to test that the workflows are working. This data is taken from the Russian News Commentary corpus. The full parallel training data can be found [here](http://www.statmt.org/wmt13/translation-task.html).

To use this data with the supervised workflow, replace the global variables for dev, test, and train in supervised.tape with the absolute paths to these files.

To use the training set with the unsupervised workflow, you first have to create a file without the preprocessing, i.e. `cat train.preprocessed.en-ru | cdec/corpus/cut-corpus.pl 1,2 > train.en-ru` and set the global variable for train in unsupervised.tape to the absolute path to this file, as well as changing the dev and test variables as above.
