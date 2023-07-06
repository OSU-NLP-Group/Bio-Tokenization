# Biomedical Language Models are Robust to Sub-optimal Tokenization

\
**Our Paper:**
- [Biomedical Language Models are Robust to Sub-optimal Tokenization](https://arxiv.org/abs/2306.17649) <br>Bernal Jiménez Gutiérrez, Huan Sun, Yu Su <br> BioNLP @ ACL 2023 


**Available HuggingFace Models:**
- [BioVocabBERT](https://huggingface.co/osunlp/BioVocabBERT)
- [PubMedBERT_Replica](https://huggingface.co/osunlp/PubMedBERT_Replica)

## Data

### SIGMORPHON Data

The datasets from the [SIGMORPHON 2022 Word-Level Morphological Segmentation Shared Task](https://github.com/sigmorphon/2022SegmentationST) 
that are relevant for our work are added to this repository directly as follows. 

```
data/eng.word.train.tsv
data/eng.word.dev.tsv
```

### UMLS

Due to the UMLS license, we are unable to distribute the MRCONSO.RRF file necessary for our work. After creating a UMLS account, 
you can download the necessary UMLS version (we use the UMLS 2022AB) from this [archive](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html). 
For reproducibility, add the MRCONSO.RRF file under the ```data``` directory before running any scripts.

### PubMed Pretraining

We use a previously processed set of PubMed abstracts made available by [BlueBERT](https://github.com/ncbi-nlp/bluebert/tree/master) from 
this [link](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/pubmed_uncased_sentence_nltk.txt.tar.gz). Download this corpus and 
add it to the ```data``` directory before starting.

### Downstream Tasks

All data for downstream performance evaluation can be obtained through the repositories used for evaluation. As explained in
more detail below, we refer to people interested in reproducing downstream results to those repositories for 
[NER](https://github.com/dki-lab/few-shot-bioIE/tree/main) 
and [entity linking](https://github.com/cambridgeltl/sapbert).

## Installation

Run the following commands to create a conda environment with the required packages. 

```
conda create -n bio-tok python=3.11 pip
conda activate bio-tok
pip install -r requirements.txt
```

## Evaluating Tokenization Quality

We use morphological segmentation as a way to evaluate how well-aligned biomedical tokenizers are to the segmentation 
judgements of biomedical expert.

### Load UMLS Phrases

We first load biomedical phrases from the UMLS that are in English as well as not suppressed by the UMLS in order to get a 
higher quality set of biomedical phrases. We run the ```src/parse_umls_file.ipynb``` Jupyter notebook to carry out this process. 
Run this notebook **only after** obtaining the MRCONSO.RRF file mentioned above.

### Create Biomedical Subset of SIGMORPHON 2022

Now that we have a higher quality set of biomedical phrases, we first extract biomedical terms and intersect them with the 
development SIGMORPHON 2022 dataset mentioned above in order to evaluation biomedical morphological segmentation quality 
specifically. The Jupyter notebook ```src/create_biomedical_sigmorphon.ipynb``` carries out 
this process and creates the file ```data/dev_bio_words.tsv``` which holds the biomedical subset used for evaluation.

### Standard Tokenizer Evaluation

We use the evaluation metrics used by the SIGMORPHON 2022 Shared Task to evaluate standard and new tokenizers. The Jupyter 
notebook ```src/evaluate_tokenizers_sigmorphon_task.ipynb``` uses the biomedical subset created above to assess tokenizer quality.

**NOTE:** The final part of this notebook requires the fine-tuned CANINE model created in the following section. Return to this
notebook after you have created the supervised model to reproduce the results in our paper.

## Creating a Well-Aligned Tokenizer 

Given the poor performance of standard tokenizer on biomedical morphological segmentation, we try to create a new, better aligned
tokenizer.

### Fine-Tuning a Character LM for Segmentation

We first fine-tune a character-based language model using the SIGMORPHON 2022 word-level segmentation training set. To this we 
run the following command:

```
CUDA_VISIBLE_DEVICES=0 python src/train_char_lm_sigmorphon.py
```

This script will fine-tune a CANINE model and create a strong supervised segmentation model. The resulting models will be 
saved under ```output/canine_exps```.

### Constructing BioVocabBERT

Now that we have a strong biomedical segmentation model, we use it to create a LM friendly tokenizer. Vocabulary free tokenizers
like our CANINE model have strong downsides such as needing a large amount of tokens for training and an out-of-vocabulary token 
for inference which could hinder a LM's generalizability. We therefore create a high quality and high coverage vocabulary of 
biomedical tokens using the following steps:
1) CANINE segments all biomedical terms in UMLS.
2) Filter only tokens which happen more than once.
3) Add back the tokens from the BERT tokenizer to cover general domain data as well.
4) Create BioVocabBERT!

We carry out these steps in the Jupyter notebook ```Create BioVocabBERT Tokenizer.ipynb```. The tokenizer will be saved as
```../output/biovocabbert_tokenizer```.

## Language Model Pre-Training

Our pre-training procedure and code is closely based on the [academic-budget-bert](https://github.com/IntelLabs/academic-budget-bert/tree/main) code-base. 
It allowed us to accelerate the pre-training process considerably. Check out their code and paper for more information on 
the techniques that enable this speedup.

### Creating the Pre-Training Data

Before starting the pre-training process, make sure that you have downloaded the corpus mentioned in the above [data](#Data) section
and have created the biovocabbert tokenizer.

We follow the original repository's procedure by running the following commands.
It first shards the pre-training corpus and then generates samples for both PubMedBERT and BioVocabBERT tokenizers.
Samples must be generated for the tokenizer of each LM you would like to pre-train. 

```
bash shard_data.sh
bash generate_sample_pubmedbert.sh #PubMedBERT Sample Creation 
bash generate_sample_biovocabbert.sh #BioVocabBERT Sample Creation
```

### Run Pre-Training

We can now finally run pre-training for both our PubMedBERT replica and the BioVocabBERT model. To start 
pre-training, run the following scripts. Make sure to adjust the ```train_micro_batch_size_per_gpu``` parameter 
depending on the size of your GPUs. 

```
bash pretrain_pubmedbert_replica.sh
bash pretrain_biovocabbert.sh
```

## Downstream Biomedical Evaluation

For evaluating our model's you can either obtain them directly from HuggingFace or follow the steps above to reproduce the 
pre-training steps.

We refer the reader interested in reproducing our downstream results to other repositories which we used directly for 
fine-tuning and evaluation purposes. Datasets can be obtained by following instructions in each repository.

For **Named Entity Recognition (NER)**, we use the [Few-Shot-BioIE](https://github.com/dki-lab/few-shot-bioIE/tree/main) repository 
to run both our fully-supervised and low-resource fine-tuning. Make sure that you set the ```kfolds``` parameter to 1 in order
to use the true development set as we have in our work.

For Entity Linking (EL), we use the [SapBERT](https://github.com/cambridgeltl/sapbert) repository. We only use the code under 
the ```evaluation``` directory which evaluates the zero-shot entity linking performance of biomedical language models.
