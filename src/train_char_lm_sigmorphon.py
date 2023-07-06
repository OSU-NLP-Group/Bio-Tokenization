#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
import re
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from glob import glob
import wandb
import random
import os
import pickle

def transform_seg_to_bio(word, segmentation, seg_pattern='@@'):
    
    word = str(word).lower()
    segmentation = str(segmentation).lower()
    
    #Handling the 'ies' plural rule
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

def get_bio_tags(df):
    bio_tags = []

    for i,row in df.iterrows():

        word = row[0]
        segmentation = row[1]

        bio = transform_seg_to_bio(word, segmentation)

        bio_tags.append(bio)
    
    df['bio'] = bio_tags
    
    return df

def encode_seg_dataframe(df, max_length=30):
    phrases = [str(w).lower() for w in df[0]]
    labels = df.bio.values

    return encode_data(phrases, labels, max_length)
    
def encode_data(texts, labels, max_length=30):
    label_to_id = {'B':0,'I':1}
    
    outputs = tokenizer.batch_encode_plus(texts, max_length=max_length,truncation=True,padding='max_length',return_tensors='pt')
    input_ids, attention_mask = outputs['input_ids'], outputs['attention_mask']
    
    labels = Tensor([[0] + [label_to_id[l] for l in lab_seq] + [0 for _ in range(max(0,max_length - (len(lab_seq)+1)))]  for lab_seq in labels]).long()
    
    dataset = TensorDataset(input_ids, attention_mask, labels)
    
    return dataset


# In[16]:


def train_epoch(model, train_data, optimizer, scheduler, epoch_num):
    print("")
    print('Training...')

    model.train()
    total_train_loss = 0
    
    # For each batch of training data...
    for step, batch in tqdm(enumerate(train_data)):
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        outputs = model(b_input_ids,
                        attention_mask=b_input_mask,
                        token_type_ids=None,
                        labels=b_labels)
        loss, logits = outputs['loss'], outputs['logits']
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_data)

    wandb.log({'avg_train_loss': avg_train_loss, 'epoch': epoch_num})

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))


# In[17]:


from datasets import ClassLabel, load_dataset

import evaluate as evaluate_hf
metric = evaluate_hf.load('seqeval')

def evaluate(model, test_data):
    print("")
    print('Evaluating...')

    model.eval()
    
    id_to_label = {0:'B',1:'I'}
    dev_loss = 0
    
    lengths = []
    all_logits = []
    all_labels = []
    
    # For each batch of training data...
    for step, batch in tqdm(enumerate(test_data)):

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
        all_labels.extend(b_labels.detach().cpu().numpy())
        
        dev_loss += loss

    all_logits = np.argmax(all_logits,axis=2)
    
    dev_gold = []
    dev_prediction = []

    for labs, preds, length in zip(all_labels, all_logits, lengths):
            
        labs = [id_to_label[l] for l in labs[1:length-1]]
        preds = [id_to_label[l] for l in preds[1:length-1]]
        
        dev_gold.append(labs)
        dev_prediction.append(preds)
    
    results = metric.compute(predictions=dev_prediction, references=dev_gold)
    
    print("")
    print("  Average dev loss: {0:.2f}".format(dev_loss / len(test_data)))
    print("  F1 Score: {0:.2f}".format(results['overall_f1']))
    results['epoch_num'] = epoch_i
    
    wandb.log(results)
    
    return results, dev_gold, dev_prediction

def seed_torch(seed=12345):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


hyperparameter_defaults = dict(
    per_device_train_batch_size=512,
    per_device_eval_batch_size=1028,
    learning_rate=1e-3,
    num_train_epochs=10,
    weight_decay=0.0,
    warmup_ratio=0.0,
    model_name="google/canine-c",
    seed=42,
    max_length=30,
    train_samples=460000
)

if __name__ == "__main__":
    wandb.init(config=hyperparameter_defaults, project="segmentation")

    seed_torch(wandb.config.seed)
    
    print(wandb.config)
    
    #Load Data
    train = pd.read_csv('../data/eng.word.train.tsv',sep='\t',header=None)
    dev = pd.read_csv('../data/eng.word.dev.tsv',sep='\t',header=None)

    train = get_bio_tags(train)
    dev = get_bio_tags(dev)

    train['len'] = [len(str(w)) for w in train[0]]
    dev['len'] = [len(str(w)) for w in dev[0]]

    print(len(train),len(dev))
    train = train[train.len < wandb.config.max_length][:wandb.config.train_samples]
    dev = dev[dev.len < wandb.config.max_length]
    print(len(train),len(dev))

    model = AutoModelForTokenClassification.from_pretrained(wandb.config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(wandb.config.model_name)

    device = 'cuda'
    model.to(device)

    optimizer = AdamW(model.parameters(),
                          lr=wandb.config.learning_rate,
                          weight_decay=wandb.config.weight_decay)
    
    epochs = wandb.config.num_train_epochs
    warmup_ratio = wandb.config.warmup_ratio
    print('epochs =>', epochs)
    
    total_steps = len(train) * epochs
    num_warmup_steps = warmup_ratio * total_steps

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    train_dataset = encode_seg_dataframe(train)
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=wandb.config.per_device_train_batch_size  # Trains with this batch size.
    )

    dev_dataset = encode_seg_dataframe(dev)
    dev_dataloader = DataLoader(
        dev_dataset,
        sampler=SequentialSampler(dev_dataset), 
        batch_size=wandb.config.per_device_eval_batch_size  # Trains with this batch size.
    )

    results_by_epoch = []
    
    exp_dirs = glob('../output/canine_exps/*')
    path = '../output/canine_exps/{}'.format(len(exp_dirs))
    os.makedirs(path)
    
    pickle.dump(dict(wandb.config),open(path+'/config.p','wb'))
    
    best_f1 = 0
        
    for epoch_i in range(1, epochs+1):
        print('======== Epoch {:} / {:} ========'.format(epoch_i, epochs))

        train_epoch(model, train_dataloader, optimizer, scheduler, epoch_i)
        results, golds, predictions = evaluate(model, dev_dataloader)
        
        results_by_epoch.append((results, golds, predictions))
        
        f1 = results['overall_f1']
        
        if f1 > best_f1:
            model.save_pretrained(path+'/best_model')
            pickle.dump(results_by_epoch,open(path+'/results.p'.format(epoch_i),'wb'))
            
            best_f1 = f1
        
    print('Done')