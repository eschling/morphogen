global {
    # train, dev and test files in parallel format:
    # SRC ||| TGT
    # Replace with your own files
    train=/home/cdyer/projects/target-morph/swahili-new/all-train.en-sw
    dev=/home/cdyer/projects/target-morph/swahili-new/dev_and_test/dev.en-sw
    test=/home/cdyer/projects/target-morph/swahili-new/dev_and_test/test.en-sw  
    # English clusters: http://www.ark.cs.cmu.edu/cdyer/en-c600.gz
    en_clusters=/usr0/corpora/wmt13/brown_clusters/en-c600.gz

    # 8-bit encoding scheme for tgt language
    # (REQUIRED FOR UNSUPERVISED SEGMENTATION)
    # (e.g. iso8859[1-16])
    encoding="iso88591"

    # For specifying your own segmentations, instead of using fast_umorph
    # analyses in fast_umorph format 
    # Format: token	prefix^prefix^<stem>^suffix^suffix
    analyses=/usr0/home/cdyer/projects/target-morph/swahili-new/types.sw.seg.1e-6_1e-4_1e-6
    
    # monolingual data for inflection training
    monolingual="/home/cdyer/projects/target-morph/swahili-new/gv+hel+dli.sw"

    # point to your morphogen  
    morphogen=/usr0/home/eschling/tools/morphogen

    # Target language model - all language models should be in KenLM format
    klm=/usr0/home/cdyer/projects/target-morph/swahili-new/lm/swahili.4gram.klm

    # Class based target language model and log probability emission map 
    # (for relating words to their cluster)
    classlm=""
    emission_map=""

    # Number of cores to use when:
    # - tuning with MIRA and decoding 
    # - extracting per-sentence grammars (lemmatized and original)
    tune_cores=10
    extract_gra_cores=10

    # Email address to send decoding results
    # (you can remove this option from the mira call in task tune_mira
    # if you would rather not get an email every time it finishes evaluating)
    email=""

    ducttape_experimental_packages=true
}

# If you have any of these tools already, change them to point to the right places
# In practice, these should be cloned locally and pointed to your local
# machine. For example, the cdec package  would then become:
# package cdec :: .versioner=disk .path=/home/eschling/cdec {}
# (there is no versioning when disk is specified)

# clone TurboParser and download trained models for tagging and parsing English
package turboparser :: .versioner=git .repo="https://github.com/andre-martins/TurboParser.git" .ref=HEAD {
  ./install_deps.sh
  ./configure && make && make install
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:`pwd;`/deps/local/lib:"
  mkdir models
  cd models
  wget http://www.ark.cs.cmu.edu/TurboParser/sd_models/210basic_sd204.tar.gz
  wget http://www.ark.cs.cmu.edu/TurboParser/sample_models/english_proj_tagger.tar.gz
  for i in *.tar.gz; do tar xvfz $i; done
  rm *.tar.gz
}

package fast_umorph :: .versioner=git .repo="https://github.com/vchahun/fast_umorph.git" .ref=HEAD {
#fast_umorph requires that the OpenFST library be installed on your machine
  make
}

package cdec :: .versioner=git .repo="git://github.com/redpony/cdec.git" .ref=HEAD 
{
  autoreconf -ifv
  # Morphogen contains code for LBFGS training with mpirun instead of SGD,
  # which can be run through cdec if it is configured with --enable-mpi
  # the maximum number of features allowed by cdec may need to be increased

  # Boost C++ libraries are required by cdec
  ./configure # --with-boost=/path/to/boost-install
  make
  ./tests/run-system-tests.pl

  # Build the python extensions
  cd python
  python setup.py build
}

# Run fast umorph to produce unsupervised segmentations
# The hyperparameters do matter, although we've found that
# alpha_stem = 1e-4 and alpha_prefix = alpha_suffix = (1e-5 | 1e-6)
# are generally a decent starting point
# the number of iterations depends on the language.
task umorph : cdec : fast_umorph :: morphogen=@
:: mono=(Mono: no=false yes=true)
:: alpha_prefix=1e-5
:: alpha_stem=1e-4
:: alpha_suffix=1e-5
:: niter=1000
:: encoding=@
< bitext=$train
< monolingual=(Mono: no="" yes=$monolingual)
> types
> segs
{
  if $mono ; then
    cat $bitext | $cdec/corpus/cut-corpus.pl 2 | cat - $mono | $cdec/corpus/lowercase.pl > temp
  else
    cat $bitext | $cdec/corpus/cut-corpus.pl 2 | $cdec/corpus/lowercase.pl > temp
  fi
  cat temp | $morphogen/scripts/extract-types.pl > types
  iconv -c -f utf8 -t $encoding types | $fast_umorph/segment $niter $alpha_prefix $alpha_stem $alpha_suffix | iconv -f $encoding -t utf8 > $segs
  rm temp
}

# EN ||| EN POS ||| EN dep ||| EN clusters
task process_src : cdec : turboparser :: morphogen=@ :: en_clusters=@
< corpus=(Section: train=$train dev=$dev test=$test)
> out {
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$turboparser/deps/local/lib:"
  paste <(cat $corpus | $cdec/corpus/cut-corpus.pl 1 |\
  $morphogen/scripts/turbotag.py $turboparser |\
  $morphogen/scripts/turboparse.py -p $turboparser)\
  <(cat $corpus | $cdec/corpus/cut-corpus.pl 1 |\
 $cdec/corpus/lowercase.pl | $morphogen/scripts/to-cluster.py $en_clusters) |\
  sed 's/	/ ||| /g' > $out
}

# TGT ||| TGT stem ||| TGT tag
task process_tgt : cdec 
:: morphogen=@ 
:: mono=(Mono: no=false yes=true)
< corpus=(Mono: no=$train yes=$monolingual)
< analyses=$segs@umorph
> out {
  if $mono; then
    cat $corpus | python $morphogen/morphogen/seg_tags.py $analyses > $out
  else  
    cat $corpus | $cdec/corpus/cut-corpus.pl 2 |\
    python $morphogen/morphogen/seg_tags.py $analyses > $out
  fi
}

# en ||| tgt stem_tag
task lemmatized_training : cdec :: morphogen=@
< src=$out@process_src[Section:train]
< tgt=$out@process_tgt[Mono:no]
> corpus {
    paste $src $tgt <(sed 's/.*//g' < $src) | sed 's/	/ ||| /g' |\
    python $morphogen/morphogen/lex_align.py --partial > $corpus
}


# Alignments (from stems)
task align : cdec
< bitext=$corpus@lemmatized_training
> alignments {
    # src ||| tgt-lem
    $cdec/word-aligner/fast_align -i $bitext -d -v > fwd_al
    $cdec/word-aligner/fast_align -i $bitext -d -v -r > bwd_al
    $cdec/utils/atools -i fwd_al -j bwd_al -c grow-diag-final-and > $alignments
    rm fwd_al bwd_al
}

# EN ||| EN POS ||| EN dep ||| EN clusters ||| TGT ||| TGT stem ||| TGT tag ||| alignments
task merge_training
< src=$out@process_src[Section:train]
< tgt=$out@process_tgt[Mono:no]
< alignments=$alignments@align
> merged {
    paste $src $tgt $alignments | sed 's/	/ ||| /g' > $merged
}

# Reverse inflection map
task reverse_map :: morphogen=@ :: mono=(Mono: no=false yes=true)
< tgt=$out@process_tgt[Mono:no]
< monolingual=(Mono: no='' yes=$out@process_tgt[Mono:yes])
> rev_map {
  if $mono ; then
    cat $tgt $monolingual | python $morphogen/morphogen/rev_map.py $rev_map
  else
    cat $tgt | python $morphogen/morphogen/rev_map.py $rev_map
  fi
}

# Train a structured inflection model using stochastic gradient descent
task struct_train :: morphogen=@ 
:: iter=10
< rev_map=@reverse_map
< merged=$merged@merge_training
> model {
   cat $merged | python $morphogen/morphogen/struct_train.py -r 0.01 -i $iter W $rev_map $model
}

# en ||| tgt
task original_training : cdec
< merged=$merged@merge_training
> corpus {
    cat $merged | $cdec/corpus/cut-corpus.pl 1,5 | $cdec/corpus/lowercase.pl > $corpus
}

# Index parallel data into suffix array
task index_training : cdec :: maxnt=(Corpus: original="" lemma="--maxnt 0")
< corpus=(Corpus: original=$corpus@original_training
                     lemma=$corpus@lemmatized_training)
< alignments=$alignments@align
> ini
> sa {
    export PYTHONPATH=`echo $cdec/python/build/lib.*`
    python -m cdec.sa.compile -b $corpus -a $alignments -c $ini -o $sa $maxnt
}

# EN ||| EN POS ||| EN dep ||| EN clusters ||| ||| ||| ||| 
task merge_empty
< src=$out@process_src
> merged {
    cat $src | sed 's/.*//g' > empty
    paste $src empty empty empty empty | sed 's/	/ ||| /g' > $merged
    rm empty
}

# Extract grammars for dev and text
task extract_gra : cdec
< corpus=(ExtractSection: dev=$dev test=$test)
< ini=$ini@index_training
> sgm
> grammars
:: cores=$extract_gra_cores {
    export PYTHONPATH=`echo $cdec/python/build/lib.*`
        cat $corpus | $cdec/corpus/lowercase.pl |\
    python -m cdec.sa.extract -c $ini -g $grammars -j $cores -z > $sgm
}

# Synthetic grammar
task synthetic_grammar :: morphogen=@
:: iter=$iter@struct_train
< rev_map=@reverse_map
< inflection_model=$model@struct_train
< corpus=(ExtractSection: dev=$merged@merge_empty[Section:dev]
                          test=$merged@merge_empty[Section:test])
< original_sgm=$sgm@extract_gra[Corpus:original]
< lemma_sgm=$sgm@extract_gra[Corpus:lemma]
> grammars
> sgm {
    cat $corpus | python $morphogen/morphogen/synthetic_grammar.py\
    $rev_map $inflection_model/model.$iter.pickle $original_sgm $lemma_sgm $grammars > $sgm
}

task make_ini :: cluster=(ClassLM: yes=true no=false)
< klm=@
< classlm=@
< emission_map=@
> cdec_ini {
    echo "formalism=scfg" >> $cdec_ini
    echo "add_pass_through_rules=true" >> $cdec_ini
    echo "feature_function=WordPenalty" >> $cdec_ini
    echo "feature_function=KLanguageModel $klm" >> $cdec_ini
    if $cluster ; then
      echo "feature_function=KLanguageModel -n ClassLM $classlm -m $emission_map" >> $cdec_ini
    fi
}

# Tune system using development data with MIRA
# then evaluate the decoder using dev and test sets.
task tune_mira : cdec 
:: cores=$tune_cores 
:: i=(MiraIter: 1 2 3)
:: email=@
< cdec_ini=@make_ini
< dev_sgm=$sgm@synthetic_grammar[ExtractSection:dev]
< test_sgm=$sgm@synthetic_grammar[ExtractSection:test]
> weights=mira/weights.final {
    $cdec/training/mira/mira.py \
      --max-iterations 20 \
      --jobs $cores \
      --kbest-size 500 \
      --update-size 500 \
      --devset $dev_sgm \
      --output-dir ./mira \
      -t $test_sgm \
      -c $cdec_ini \
      #-e $email
}

plan basic {
  reach tune_mira via (ClassLM: no) * (Mono: no) * (MiraIter: 1)
}

plan classlm {
  reach tune_mira via (ClassLM: yes) * (Mono: no) * (MiraIter: 1)
}

plan mono {
  reach tune_mira via (ClassLM: no) * (Mono: yes) * (MiraIter: 1)
}
