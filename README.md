# Chinese Essay Dataset For Discourse Element Identification

This project includes the dataset and source code for the paper [Discourse Self-Attention for Discourse Element Identification in Argumentative Student Essays](https://www.aclweb.org/anthology/2020.emnlp-main.225/).
Detail informatin please refers to the paper.

# Dataset

    ./data/Ch_train.json    # Chinese trianing dataset

    ./data/Ch_test.json     # Chinese test dataset

    ./data/En_train.json    # English training dataset

    ./data/En_test.json     # English test dataset


# Word embeddings

    ./embd/new_embeddings2.txt  #Tencent pre-trained word embeddings from https://ai.tencent.com/ailab/nlp/en/embedding.html

    ./embd/bert-base-uncased-word_embeddings.pkl  # BERT-base-uncased word embeddings from the embeddings module of bert-base-uncased


# Training code

    ./train.py      # training code for Chinese dataset

    ./train_e.py    # training code for English dataset

    ./train_ef.py   # training code for English dataset with features


# Test code

    ./test.py

    # Chinese_test(model_dir)     for Chinese test

    # English_test(model_dir)     for English test

    # English_test_ft(model_dir)  for English test with features

# Models
    ./model/roles/DiSA_ch/DiSA_ch.pk    # The DiSA model for Chinese dataset
    
    ./model/e_roles_4/DiSA_en4/DiSA_en4.pk  # TheDiSA model for English dataset(4 classes)
    
    ./model/e_roles_3/DiSA_en3/DiSA_en3.pk  # The DiSA model for English dataset(3 classes)
    
    ./model/e_roles_3/DiSA_ft_en3/DiSA_ft_en3.pk  # The DiSA model with features for English dataset(3 classes)

# Sentence Positional Ecodings setting

parameter 'p_embd' setting

- 'add'       RelativeSPE
- 'embd_b'    PosEmbedding
- 'embd_c'    Sinusoidal


## Reference
The code and dataset are released with this paper:
```bibtex
        @inproceedings{song-etal-2020-discourse,
            title = "Discourse Self-Attention for Discourse Element Identification in Argumentative Student Essays",
            author = "Song, Wei  and
              Song, Ziyao  and
              Fu, Ruiji  and
              Liu, Lizhen  and
              Cheng, Miaomiao  and
              Liu, Ting",
            booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
            month = nov,
            year = "2020",
            publisher = "Association for Computational Linguistics",
            url = "https://www.aclweb.org/anthology/2020.emnlp-main.225",
            pages = "2820--2830",
        }
```
