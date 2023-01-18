#! /usr/bin/bash
set -e

data_path="/home/robin/data/robin/data"

echo $data
# for type in wordpiece bpe-straight spe-bpe unigram;
for type in unigram;
do
    echo $type
    # for size in 32000 64000;
    for size in 32000;
    do
        echo $size
        # for data in sample.txt sample-10.txt sample-10-again.txt;
        for data in sample-10-again.txt;
        do
            echo $data
            ls $data_path/$data
            # python train_tokenizer.py --infiles $data_path/$data \
            #                           --tokenizer_name tokenizers/tok-$data-$type-$size-pretok.json \
            #                           --vocab_size $size \
            #                           --add_prefix_space \
            #                           --tok_type $type \
            #                           --all_pts

            # echo "tokenizers/tok-$data-$type-$size-pretok.json sample.txt"
            # python test_tokenizer.py --tokenizer tokenizers/tok-$data-$type-$size-pretok.json --text $data_path/sample.txt --lines 1000 --seed 666
            # echo "tokenizers/tok-$data-$type-$size-pretok.json sample-10.txt"
            # python test_tokenizer.py --tokenizer tokenizers/tok-$data-$type-$size-pretok.json --text $data_path/sample-10.txt --lines 1000 --seed 666
            # echo "tokenizers/tok-$data-$type-$size-pretok.json sample-10-again.txt"
            # python test_tokenizer.py --tokenizer tokenizers/tok-$data-$type-$size-pretok.json --text $data_path/sample-10-again.txt --lines 1000 --seed 666

            python train_tokenizer.py --infiles $data_path/$data \
                                      --tokenizer_name tokenizers/tok-$data-$type-$size-straight.json \
                                      --vocab_size $size \
                                      --add_prefix_space \
                                      --tok_type $type

            echo "tokenizers/tok-$data-$type-$size-straight.json sample.txt"
            python test_tokenizer.py --tokenizer tokenizers/tok-$data-$type-$size-straight.json --text $data_path/sample.txt --lines 1000 --seed 666
            echo "tokenizers/tok-$data-$type-$size-straight.json sample-10.txt"
            python test_tokenizer.py --tokenizer tokenizers/tok-$data-$type-$size-straight.json --text $data_path/sample-10.txt --lines 1000 --seed 666
            echo "tokenizers/tok-$data-$type-$size-straight.json sample-10-again.txt"
            python test_tokenizer.py --tokenizer tokenizers/tok-$data-$type-$size-straight.json --text $data_path/sample-10-again.txt --lines 1000 --seed 666
            
            echo ""
        done
        echo ""
    done
    echo ""
done
