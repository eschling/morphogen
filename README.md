`morphogen` is a tool for improving machine translation into morphologically rich languages. It uses source context to predict inflections, and uses these inflection models to create augmented grammars that can be used with a standard decoder.

A paper describing this tool, which appeared in the Prague Bulletin of Mathematical Linguistics, can be found [here](http://ufal.mff.cuni.cz/pbml/100/art-schlinger-chahuneau-dyer.pdf). This work and [fast_umorph](http://github.com/vchaun/fast_umorph) will also be presented at EMNLP 2013.

## Dependencies

Example workflows for using morphogen are provided using [ducttape](https://github.com/jhclark/ducttape). If you have ducttape and morphogen installed, the ducttape workflows will install all of the external programs for you. However, this assumes that you have the dependencies for the external tools already installed on your system (e.g. Boost for cdec, OpenFST for fast_umorph, etc.). While we try to make this process as painless as possible, we cannot anticipate all problems with external tools.

While the morphogen code itself is not dependent on anything external, it is intended to be used with a number of external tools. Specifically, it is used to extend the per-sentence grammars created by [cdec](http://www.cdec-decoder.org). The inflection model depends on having good source side information, in the form of dependency parsing, part-of-speech tagging, and word clustering. We do these using [TurboParser](http://www.ark.cs.cmu.edu/TurboParser/), TurboTagger, and [600 Brown clusters](http://www.ark.cs.cmu.edu/cdyer/en-c600.gz) produced from large amounts of monolingual English data. These are all publically available. 

If no morphological segmentations are given, we use [fast_umorph](https://github.com/vchahun/fast_umorph) to get unsupervised morphological segmentations. This requires the [OpenFST library](http://www.openfst.org/) to be installed and in your path. The Makefile for fast_umorph assumes g++ 4.7. There is also a known issue with compiling and OpenFST (more info [here](https://github.com/vchahun/fast_umorph/issues/1))

The tagging and parsing could theoretically be done with any tool. Morphogen only requires that the dependency parses are in the Stanford dependency format.

`morphogen/structlearn` contains scripts for training the model with stochastic gradient descent implemented in [Cython](cython.org) and Python. It is strongly recommended that you install Cython, as it takes significantly longer to train the inflection models using the standard Python implementation. 

## Running

### Unsupervised

A [ducttape](https://github.com/jhclark/ducttape) workflow (unsupervised.tape) is provided in the examples folder. If you replace the dev, test, and train variables in unsupervised.tape with paths to your data sets (in ` SRC ||| TGT ` bitext format), point the morphogen global variable at your clone of morphogen, specify the 8-bit encoding for your target language, and run `ducttape unsupervised.tape -p basic -O unsup` it will:
- clone all of the necessary external tools to your machine
- preprocess your data
- produce unsupervised morphological segmentations with [fast_umorph](https://github.com/vchahun/fast_umorph)(this can take time, depending on the number of iterations)
- use them to train an inflection model with stochastic gradient descent
- extract per-sentence-grammars for the dev and test sets with cdec
- augment your grammars with the inflection model
- tune your augmented system using MIRA
- evaluate your system on the given test set

The `-p basic` option says that we should run the workflow path `basic`. The `-O` option specifies the output directory for the workflow.

There is a very good, if incomplete, ducttape tutorial [here](http://nschneid.github.io/ducttape-crash-course/tutorial.html)

If you have any of the dependencies already installed (cdec, TurboParser), you can point the ducttape workflows to these and use them instead. There is an example in the ducttape files.

The corpus used with fast_umorph to get unsupervised segmentations must be encoded in an **8-bit format** (e.g. ISO-8859-[1-16]). If you're using our `unsupervised.tape`, this means you need to specify the correct 8-bit encoding for your target language in the global variables. The script will do the necessary conversions for you. 

The unsupervised morphological segmentations take three hyperparameters (`alpha_prefix`, `alpha_stem`, and `alpha_suffix`). We have found that `alpha_prefix, alpha_suffix << alpha_stem << 1` is necessary to produce useful segmentations. This encodes that there should be many more possible stems than there are inflectional affixes. The number of iterations necessary to produce good segmentations varies depending on the language. In general `alpha_prefix = alpha_suffix = 1e-6` , `alpha_stem = 1e-4` at 1000 iterations is a good starting point. 

You can edit the ducttape file to specify your own segmentations. Simply define a global variable that points to your segmentations and replace all references to the output of the umorph task with your variable. You do not need to remove the task. It will not run if it is not required to reach the end goal of the plan.

Format: 
token &emsp; prefix^prefix^&lt;stem&gt;^suffix^suffix
(e.g. тренинговой &emsp; <тренинг>^ов^ой)

### Supervised

We also provide a ducttape workflow for supervised models, and an example configuration file for a Russian positional tagset. If you will be training a supervised model, all of the target side preprocessing must be done by you. 
Format: ` SRC ||| TGT ||| TGT stem ||| TGT tag `

`struct_train.py` assumes that the first letter of the tag is the word category, but this can be easily modified.

Similarly, any monolingual data must be preprocessed.
Format: ` TGT ||| TGT stem ||| TGT tag `
        
You MUST provide a Python configuration file which defines a function `get_attributes(category, attributes)`, which yields a list of features given the category and morphological analysis of a word. This file must be placed in the folder `morphogen/config_files/`. Since this function depends on the format produced by your supervised morphological analyzer, you must define it yourself. This will be used when training the inflection model to create sparse vectors of target inflectional features. There is an example function for a Russian positional tagset in `morphogen/config_files/russian_config.py`.

### Both

If you don't specifiy a target language model, a 4-gram language model will be created as a part of the ducttape workflow. We recommend also creating a class based target language model and using this in addition to the standard target language model. Our class based language models were created from 600 brown clusters trained on monolingual data and smoothed with Witten-Bell. All language models must be in the KenLM format for use with cdec

We also provide a ducttape script for intrinsic evaluation of inflection models. This preprocesses the given development data in the same manner as the training data and evaluates our hypothesized inflections against the actual inflections. It's more of a sanity check than anything else.

To inspect the feature weights learned by a given model, use `python show_model.py model > model.features`

## Current Work

An implementation of [Adagrad](http://www.cs.berkeley.edu/~jduchi/projects/DuchiHaSi10.pdf) (with or without L1 regularization) has been added but is not fully tested (use at your own risk). 

## License

Copyright © 2013 Victor Chahuneau, Eva Schlinger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
 
