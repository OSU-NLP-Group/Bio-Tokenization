from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from datasets import ClassLabel, load_dataset

import pandas as pd
import numpy as np
from tqdm import tqdm
import re

from tokenizers.pre_tokenizers import Sequence, Whitespace, Punctuation, Split
whitespace_pretokenizer = Sequence([Whitespace(), Punctuation()])

def transform_seg_to_bio(word, segmentation, seg_pattern='@@'):
    
    word = str(word).lower()
    segmentation = str(segmentation).lower()
    
    #Handling the 'ies' plural rule
    if seg_pattern=='@@':
        segmentation = re.sub('y @@s$',' @@ie @@s',segmentation)
    
    segmented_tokens = segmentation.replace(seg_pattern,'').split()
    
    if ''.join(segmented_tokens) != word:
        
        segmented_word = word

        #Last Token
        token = segmented_tokens[-1]
        if segmented_word.endswith(token):
            segmented_word = segmented_word[:-len(token)] + ''.join([str(len(segmented_tokens)) for _ in token])

        #First Token
        token = segmented_tokens[0]
        if segmented_word.startswith(token):
            segmented_word = ''.join([str(len(segmented_tokens)) for _ in token]) + segmented_word[len(token):]
        
        #Rest of Tokens
        for i,token in enumerate(segmented_tokens[1:-1]):
            segmented_word = segmented_word.replace(token,''.join([str(i+1) for _ in token]))
    
        tokenized = []
        curr_token = []
        
        prev_id = 0
        prev_number = False
        
        for curr_id,letter in zip(segmented_word, word):
            try:
                curr_id = int(curr_id)
                number = True
            except:
                number = False
                
            if curr_id != prev_id and (prev_number or number): 
                if len(curr_token) > 0:
                    tokenized.append(''.join(curr_token))
                    curr_token = []
            
            curr_token.append(letter)
            
            prev_id = curr_id
            prev_number = number
        
        tokenized.append(''.join(curr_token))
    else:
        tokenized = segmented_tokens
        
    bio = []
    
    for token in tokenized:
        bio.append('B' + ''.join(['I' for _ in token[1:]]))
        
    bio = ''.join(bio)
    
    assert len(bio) == len(word),ipdb.set_trace()
    return bio


def get_bio_tags(df, word_col=0, seg_col=1):
    bio_tags = []

    for i,row in df.iterrows():

        word = row[0]
        segmentation = row[1]

        bio = transform_seg_to_bio(word, segmentation)

        bio_tags.append(bio)
    
    df['bio'] = bio_tags
    
    return df

def get_bio_tag_col(words, segs, segmentation='##'):
    
    bio_tags = []

    for word, segmentation in zip(words,segs):

        bio = transform_seg_to_bio(word, segmentation)

        bio_tags.append(bio)
        
    return bio_tags

def bio_to_tokenized(word, bio):
    
    tokenized = []
    
    curr_token = []
    
    prev_tag = 'B'
    
    index = 0
    
    for letter, tag in zip(word, bio):
        
        if tag == 'B' and len(curr_token) > 0:
            if index == 0:
                tokenized.append(''.join(curr_token))
            else:
                tokenized.append('##' + ''.join(curr_token))
                
            curr_token = []
            index += 1
            
        curr_token.append(letter)
        prev_tag = tag
        
    if index == 0:
        tokenized.append(''.join(curr_token))
    else:
        tokenized.append('##' + ''.join(curr_token))
        
    return ' '.join(tokenized)
        
def tokenize_word_from_bio(words, bio_tags):
    
    tokenized_words = []

    for word, bio in zip(words, bio_tags):

        tokenized_word = bio_to_tokenized(word,bio)
        tokenized_words.append(tokenized_word)
    
    return tokenized_words


def encode_data(texts, labels, max_length=30):
    ##Encode data using character based LM
    tokenizer = AutoTokenizer.from_pretrained('google/canine-c')

    label_to_id = {'B':0,'I':1}
    
    outputs = tokenizer.batch_encode_plus(texts, max_length=max_length,truncation=True,padding='max_length',return_tensors='pt')
    input_ids, attention_mask = outputs['input_ids'], outputs['attention_mask']
    
    labels = Tensor([[0] + [label_to_id[l] for l in lab_seq] + [0 for _ in range(max(0,max_length - (len(lab_seq)+1)))]  for lab_seq in labels]).long()
    
    dataset = TensorDataset(input_ids, attention_mask, labels)
    
    return dataset

def tokenize_phrase(phrase, model,device='cuda'):
     
        
    model.to(device)
    tokens = [t[0] for t in whitespace_pretokenizer.pre_tokenize_str(phrase)]
    
    id_to_label = {0:'B',1:'I'}

    dataset = encode_data(tokens, [['I' for l in token] for token in tokens])
    dataloader = DataLoader(
    dataset,
    sampler=SequentialSampler(dataset), 
    batch_size=16)
    
    lengths = []
    all_logits = []
    all_labels = []
    
    # For each batch of training data...
    for step, batch in enumerate(dataloader):

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                            token_type_ids=None,
                            labels=b_labels)
            loss, logits = outputs['loss'], outputs['logits']
        
        lengths.extend(b_input_mask.sum(dim=1).detach().cpu().numpy())
        all_logits.extend(logits.detach().cpu().numpy())
    
    all_logits = np.argmax(all_logits,axis=2)
    
    bios = []

    for preds, length in zip(all_logits, lengths):
            
        preds = [id_to_label[l] for l in preds[1:length-1]]
        bios.append(preds)
    
    tokenized_words = tokenize_word_from_bio(tokens, bios)
    
    return ' '.join(tokenized_words)

def supervised_tokenized_phrase(phrase, model, device='cuda'):
    
    model.to(device)
    model.eval()
    
    tokens = [t[0] for t in whitespace_pretokenizer.pre_tokenize_str(phrase)]
    
    id_to_label = {0:'B',1:'I'}

    dataset = encode_data(tokens, [['I' for l in token] for token in tokens])
    dataloader = DataLoader(
    dataset,
    sampler=SequentialSampler(dataset), 
    batch_size=16)
    
    lengths = []
    all_logits = []
    all_labels = []
    
    with torch.no_grad():

        # For each batch of training data...
        for step, batch in enumerate(dataloader):

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                            token_type_ids=None,
                            labels=b_labels)
            loss, logits = outputs['loss'], outputs['logits']

            lengths.extend(b_input_mask.sum(dim=1).detach().cpu().numpy())
            all_logits.extend(logits.detach().cpu().numpy())

    all_logits = np.argmax(all_logits,axis=2)
    
    bios = []

    for preds, length in zip(all_logits, lengths):
            
        preds = [id_to_label[l] for l in preds[1:length-1]]
        bios.append(preds)
    
    tokenized_words = tokenize_word_from_bio(tokens, bios)
    
    return tokenized_words

def tokenize_long_list(long_word_list, best_model):
    tokenized_words = []

    max_batch = 500

    batch = []
    for word in tqdm(long_word_list):

        if len(batch) < max_batch:
            batch.append(word)
        else:
            tokenized_words.extend(supervised_tokenized_phrase(' '.join(batch), best_model))
            batch = [word]

    tokenized_words.extend(supervised_tokenized_phrase(' '.join(batch), best_model))
    
    return tokenized_words