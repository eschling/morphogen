## Examples

unsupervised.tape
supervised.tape
intrinsic.tape

## Data

The data folder contains our russian dev and test sets, and a small portion of our supervised russian training set (3000 sentences), to test that the workflows are working. This data is taken from the Russian News Commentary corpus. The full parallel training data can be found [here](http://www.statmt.org/wmt13/translation-task.html).

To use this data with the supervised workflow, replace the global variables for dev, test, and train in supervised.tape with the absolute paths to these files.

To use the training set with the unsupervised workflow, you first have to create a file without the preprocessing, i.e. `cat train.preprocessed.en-ru | cdec/corpus/cut-corpus.pl 1,2 > train.en-ru` and set the global variable for train in unsupervised.tape to the absolute path to this file, as well as changing the dev and test variables as above.
