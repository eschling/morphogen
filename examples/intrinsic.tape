# Workflow for performing intrinsic evaluation of inflection prediction model
global {
    dev=/usr0/home/cdyer/projects/target-morph/swahili-new/dev_and_test/dev.en-sw
    train=/usr0/home/cdyer/projects/target-morph/swahili-new/all-train.en-sw

    # available online at http://www.ark.cs.cmu.edu/cdyer/en-c600.gz
    en_clusters=/usr0/corpora/wmt13/brown_clusters/en-c600.gz

    # model and segmentations on which you want to evaluate the accuracy
    model="/usr1/home/eschling/morph/swahili-new/clean/struct_train/Baseline.baseline/model/model.10.pickle"
    analyses=/home/cdyer/projects/target-morph/swahili-new/types.sw.seg.1e-6_1e-4_1e-6

    morphogen=/home/eschling/tools/morphogen
    turboparser=/usr1/home/eschling/morph/swahilii-new/example/.packages/turboparser/452a4ecfd19d8af89de8d2ce420e21bab8d41577/

    ducttape_experimental_packages=true
}

package cdec :: .versioner=disk .path=/home/eschling/cdec {}

# EN ||| EN POS ||| EN dep ||| EN clusters
task process_src : cdec :: turboparser=@ :: en_clusters=@ :: morphogen=@
< corpus=(Section: train=@ dev=@)
> out {
    paste <(cat $corpus | $cdec/corpus/cut-corpus.pl 1 | $morphogen/scripts/turbotag.py $turboparser | $morphogen/scripts/turboparse.py -p $turboparser)\
    <(cat $corpus | $cdec/corpus/cut-corpus.pl 1 | $cdec/corpus/lowercase.pl | $morphogen/scripts/to-cluster.py $en_clusters) |\
    sed 's/	/ ||| /g' > $out
}

# TGT ||| TGT stem ||| TGT tag
task process_tgt : cdec :: morphogen=@ :: analyses=@
< corpus=(Section: train=@ dev=@)
> out {
    cat $corpus | $cdec/corpus/cut-corpus.pl 2 |\
    python $morphogen/morphogen/seg_tags.py $analyses > $out
}

# en ||| tgt stem_tag
task lemmatized_training : cdec :: morphogen=@
< src=(Section: train=$out@process_src[Section:train] dev=$out@process_src[Section:dev])
< tgt=(Section: train=$out@process_tgt[Section:train] dev=$out@process_tgt[Section:dev])
> combine
> corpus {
    paste $src $tgt <(sed 's/.*//g' < $src) | sed 's/	/ ||| /g' > $combine
    cat $combine | python $morphogen/morphogen/lex_align.py --partial > $corpus
}

# Alignments (from stems)
task align : cdec
< bitext_train=$corpus@lemmatized_training[Section:train]
< bitext_dev=$corpus@lemmatized_training[Section:dev]
> alignments {
    len=$(wc -l < $bitext_dev)
    echo $len
    cat $bitext_train $bitext_dev > bitext
    # src ||| tgt-lem
    $cdec/word-aligner/fast_align -i bitext -d -v > fwd_al
    $cdec/word-aligner/fast_align -i bitext -d -v -r > bwd_al
    $cdec/utils/atools -i fwd_al -j bwd_al -c grow-diag-final-and | tail -$len > $alignments
    rm fwd_al bwd_al bitext
}

# EN ||| EN POS ||| EN dep ||| EN clusters ||| TGT ||| TGT stem ||| TGT tag ||| alignments
task merge_training
< src=$out@process_src[Section:train]
< tgt=$out@process_tgt[Section:dev]
< dev=@
< alignments=@align 
> merged {
    paste $src $tgt $alignments | sed 's/	/ ||| /g' > $merged
}

# Reverse inflection map
task reverse_map :: morphogen=@
< tgt=$out@process_tgt[Section:dev]
> rev_map {
    cat $tgt | python $morphogen/morphogen/rev_map.py $rev_map
}

task predict :: morphogen=@ :: ambig=(Ambiguous: no='' yes='--ambiguous') 
< rev_map=@reverse_map
< model=@
< corpus=$merged@merge_training
> out {
  cat $corpus | python $morphogen/morphogen/predict.py $rev_map $model $ambig > out
}

plan eval {
  reach predict via (Ambiguous: *)
}
