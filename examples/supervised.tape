global {
    # held out dev and test
    # Format: SRC ||| TGT
    dev=/usr0/home/cdyer/projects/target-morph/russian/dev.tokenized.en-ru
    test=/usr0/home/cdyer/projects/target-morph/russian/test.tokenized.en-ru
    # corpus for training, preprocessed on the target side
    # Format: SRC ||| TGT ||| TGT lemma ||| TGT tag
    train=/usr1/home/eschling/morph/russian/nc.clean.en-ru
    
    # tags for which to extract inflections and create inflection models
    tags='N|V|A|R|M'

    # name of config module which defines parsing for supervised tagset
    # MUST BE IN config_files folder in morphogen
    config=russian_config

    # optional monolingual data for inflection training (branch point Mono)
    # must be preprocessed in the same manner as the training data
    # Format: TGT ||| TGT lemma ||| TGT tag
    monolingual=""
    
    en_clusters=/usr0/corpora/wmt13/brown_clusters/en-c600.gz
    
    morphogen=/home/eschling/tools/morphogen/

    # optional class based language model and log probability emission map
    # (select with branch point ClassLM)
    classlm=/usr1/home/wammar/monolingual/plain-ru/all-wmt-mono.class.order7.wbdiscount.ken-binary
    emission_map=/home/wammar/russian-mt-blitz-2013/to-victor/ru-c600.emission-logprobs
    
    # Target language model - all language models must be in KenLM format
    # use branch point KLM to indicate whether you are using your own
    # target lm or making one from the training data.
    klm="/usr1/home/vchahune/wmt13.en-ru/data/news+nc+yandex+cleancrawl.ru.4.klm"
    # If you do not want to provide a target language model, one will be 
    # created from the training data and any monolingual data provided
    # (specify with branch point KLM)
    lm_order=4

    # email to send decoding results
    # this option can be added/removed from the mira.py call in the tune_mira
    # task if you would rather not receive an email when decoding has finished
    email=""

    tune_cores=10
    extract_gra_cores=10

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

# EN ||| EN POS ||| EN dep ||| EN clusters
task process_src : cdec : turboparser 
:: morphogen=@ :: en_clusters=@
< corpus=(Section: train=$train dev=$dev test=$test)
> out {
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$turboparser/deps/local/lib:"
  (cat $corpus | $cdec/corpus/cut-corpus.pl 1 | $morphogen/scripts/turbotag.py $turboparser |\
  $morphogen/scripts/turboparse.py -p $turboparser) > parsed
  cat $corpus | $cdec/corpus/cut-corpus.pl 1 | $cdec/corpus/lowercase.pl |\
  $morphogen/scripts/to-cluster.py $en_clusters > cluster
  paste parsed cluster | sed 's/	/ ||| /g' > $out
  rm parsed cluster
}

# TGT ||| TGT stem ||| TGT tag
task process_tgt : cdec
< corpus=$train
> out {
    cat $corpus | $cdec/corpus/cut-corpus.pl 2,3,4 > $out
}

# en ||| tgt stem_tag
task lemmatized_training : cdec 
:: morphogen=$morphogen 
:: tags=@
< src=$out@process_src[Section:train]
< tgt=$out@process_tgt
> corpus {
    paste $src $tgt <(sed 's/.*//g' < $src) | sed 's/	/ ||| /g' | python $morphogen/morphogen/lex_align.py --partial --tags $tags > $corpus
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
< tgt=$out@process_tgt
< alignments=@align 
> merged {
    paste $src $tgt $alignments | sed 's/	/ ||| /g' > $merged
}

# Reverse inflection map
# Monolingual data is optional. This will allow for the model to possibly
# produce inflected forms that do no appear anywhere in the training data.
task reverse_map :: morphogen=$morphogen :: tags=$tags
:: mono=(Mono: no=false yes=true)
< tgt=$out@process_tgt
< monolingual=(Mono: no="" yes=$monolingual)
> rev_map {
  if $mono ; then
    cat $tgt $monolingual | python $morphogen/morphogen/rev_map.py $rev_map -t ${tags//|/}
  else
    cat $tgt | python $morphogen/morphogen/rev_map.py $rev_map -t ${tags//|/}
  fi
}

# Train structured inflection models for each tag using stochastic gradient descent
# the model is saved after each iteration of SGD
# to conserve space, you may want to remove the initial iterations 
task struct_train :: morphogen=@
:: tags=$tags
:: iter=10
:: rate=0.01 
:: config=$config
< rev_map=@reverse_map
< merged=@merge_training
> model {
  temp=$IFS; IFS='|'; taglist=($tags); IFS=$temp
  for tag in "${taglist[@]}"; do
    if [ ! -d $model ]; then
      mkdir $model
    fi
    cat $merged | python $morphogen/morphogen/struct_train.py -c $config -r $rate -i $iter $tag $rev_map $model/$tag
  done
}

# en ||| tgt
task original_training : cdec
< merged=@merge_training
> corpus {
    cat $merged | $cdec/corpus/cut-corpus.pl 1,5 | $cdec/corpus/lowercase.pl > $corpus
}

# Index parallel data into suffix array
task index_training : cdec :: maxnt=(Corpus: original="" lemma="--maxnt 0")
    < corpus=(Corpus: original=$corpus@original_training
                         lemma=$corpus@lemmatized_training)
    < alignments=@align
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
task synthetic_grammar 
:: morphogen=$morphogen
:: tags=@
:: iter=$iter@struct_train
    < model=$model@struct_train
    < rev_map=@reverse_map
    < corpus=(ExtractSection: dev=$merged@merge_empty[Section:dev]
                             test=$merged@merge_empty[Section:test])
    < original_sgm=$sgm@extract_gra[Corpus:original]
    < lemma_sgm=$sgm@extract_gra[Corpus:lemma]
    > grammars
    > sgm {
    temp=$IFS; IFS='|'; taglist=($tags); IFS=$temp
    models=""
    for t in "${taglist[@]}"; do
      models+="$model/$t/model.$iter.pickle "
    done
    cat $corpus | python $morphogen/morphogen/synthetic_grammar.py\
    $rev_map $models $original_sgm $lemma_sgm $grammars > $sgm
}

task train_lm : cdec :: lm_order=@
:: mono=(Mono: no=false yes=true)
< bitext=$train
< monolingual=(Mono: no='' yes=$monolingual)
> arpa
> klm {
  if $mono ; then
    cat $bitext | $cdec/corpus/cut-corpus.pl 1 | cat - $monolingual |\
        $cdec/corpus/lowercase.pl |\
        $cdec/klm/lm/builder/builder --order $lm_order > $arpa
  else
    cat $bitext | $cdec/corpus/cut-corpus.pl 1 | $cdec/corpus/lowercase.pl |\
      $cdec/klm/lm/builder/builder --order $lm_order > $arpa
  fi
  $cdec/klm/lm/build_binary $arpa $klm
}

task make_ini
:: cluster=(ClassLM: yes=true no=false)
< klm=(KLM: lm=$klm@train_lm user=$klm )
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

# Tune system using development data with MIRA and evaluate on
# given test set
task tune_mira : cdec 
:: cores=$tune_cores 
:: iter=(MiraIter: 1 2 3)
:: email=@
    < cdec_ini=@make_ini
    < dev_sgm=(Method: baseline=$sgm@extract_gra[ExtractSection:dev,Corpus:original]
                      synthetic=$sgm@synthetic_grammar[ExtractSection:dev])
    < test_sgm=(Method: baseline=$sgm@extract_gra[ExtractSection:test,Corpus:original]
                        synthetic=$sgm@synthetic_grammar[ExtractSection:test])
    > weights=mira/weights.final {
    export PYTHONPATH=`echo $cdec/python/build/lib.*`
    $cdec/training/mira/mira.py \
      --max-iterations 20 \
      --optimizer 2 \
      --jobs $cores \
      --kbest-size 500 \
      --update-size 500 \
      --step-size 0.01 \
      --metric-scale 1 \
      --devset $dev_sgm \
      --output-dir ./mira \
      -t $test_sgm \
      -c $cdec_ini \
      #-e #email
}

plan test {
  reach merge_empty via (Section: test)
}

plan basic {
  reach tune_mira via (Method: synthetic) * (Mono: no) * (ClassLM: no) * (MiraIter: *)
}

plan classlm {
  reach tune_mira via (Method: synthetic) * (Mono: no) * (ClassLM: yes) * (MiraIter: *)
}

plan monolingual {
  reach tune_mira via (Method: synthetic) * (Mono: yes) * (ClassLM: no) * (MiraIter: *)
}
