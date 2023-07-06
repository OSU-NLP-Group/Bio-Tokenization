import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tokenizers import Tokenizer
import numpy as np
import re
import os
import pickle
import math
import morfessor

import ipdb

from tokenizers.pre_tokenizers import Sequence, Whitespace, Punctuation

whitespace_pretokenizer = Sequence([Whitespace(),Punctuation()])

import nltk
import re
from nltk.corpus import stopwords
stopwords = set(stopwords.words())

class MorfessorTokenizer():

    def __init__(self, tok_type, tokenizer_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', oov=True):
        morfessor_models = pickle.load(open('morfessor_tokenizers.p','rb'))
        self.morf_tok = morfessor_models[tok_type]

        self.secondary_tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.vocab = self.secondary_tokenizer.vocab

        self.oov = oov

    def tokenize(self, text):
        
        return self.augmented_tokenize_morfessor(text, self.morf_tok, self.secondary_tokenizer, self.oov)

    def augmented_tokenize_morfessor(self, text, new_tokenizer, og_tokenizer, oov):
        tokenized_with_spans = whitespace_pretokenizer.pre_tokenize_str(text)
        tokens = [t[0] for t in tokenized_with_spans]
        final_tokens = []

        for word in tokens:
            try:
                assert word not in stopwords and re.match('^[A-Za-z]*$', word)
                encoded = new_tokenizer.viterbi_segment(word)[0]
            except:
                encoded = [word]

            if len(encoded) == 1:
                final_tokens.extend(og_tokenizer.tokenize(word))
            else:
                if oov:
                    for i, part in enumerate(encoded):
                        if i > 0:
                            part = '##' + part
                        final_tokens.append(part)
                else:
                    subwords = []
                    for i, part in enumerate(encoded):
                        if i > 0:
                            part = '##' + part

                        if part in self.vocab:
                            subwords.append(part)

                    if len(subwords) ==  len(encoded):
                        final_tokens.extend(subwords)
                    else:
                        final_tokens.extend(og_tokenizer.tokenize(word))

        return final_tokens

    def convert_tokens_to_ids(self,tokens):

        return self.secondary_tokenizer.convert_tokens_to_ids(tokens)
