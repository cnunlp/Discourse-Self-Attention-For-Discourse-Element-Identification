# Chinese-Essay-Dataset-For-Sentence-Function-Identification
A Chinese argumentative student essay dataset for Sentence Function Identification.

Data
./data/Ch_train.json    Chinese trianing dataset
./data/Ch_test.json     Chinese test dataset
./data/En_train.json    English training dataset
./data/En_test.json     English test dataset


Word embeddings
./embd/new_embeddings2.txt  
Tencent pre-trained word embeddings
from https://ai.tencent.com/ailab/nlp/en/embedding.html

./embd/bert-base-uncased-word_embeddings.pkl    
BERT-base-uncased word embeddings
from the embeddings module of bert-base-uncased


Training code
./train.py      training code for Chinese dataset
./train_e.py    training code for English dataset
./train_ef.py   training code for English dataset with features


Test code
./test.py
Chinese_test(model_dir)     for Chinese test
English_test(model_dir)     for English test
English_test_ft(model_dir)  for English test with features


Sentence Positional Ecodings setting
parameter 'p_embd' setting
'add'       RelativeSPE
'embd_b'    PosEmbedding
'embd_c'    Sinusoidal