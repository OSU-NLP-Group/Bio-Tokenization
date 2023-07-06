python shard_data_no_sentence_splitting.py \
    --dir ../../data/pubmed_sentence_nltk.txt \
    -o data/pubmed_shards \
    --num_train_shards 256 \
    --num_test_shards 128 \
    --frac_test 0.1