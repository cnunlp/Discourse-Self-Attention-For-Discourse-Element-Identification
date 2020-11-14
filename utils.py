import json
import os, sys
import numpy as np
import math
from collections import Counter

import random
# random.seed(312)

UNKNOWN = [0]
PADDING = [0]
LABELPAD = 7

embd_name = ['embd', 'embd_a', 'embd_b', 'embd_c'] 

label_map = {'Introduction': 0, 
             'Thesis': 1, 
             'Main Idea': 2,
             'Evidence': 3,   
             'Conclusion': 4, 
             'Other': 5, 
             'Elaboration': 6,
             'padding': 7}

def getRelativePos(load_dict):
    gid = load_dict['gid']
    load_dict['gpos'] = [i/len(gid) for i in gid]
    lid = load_dict['lid']
    lpos = []
    temp = []
    for i in lid:
        if i == 1 and len(temp) > 0 :
            lpos += [i/len(temp) for i in temp]
            temp = []
        temp.append(i)
    if len(temp) > 0:
        lpos += [i/len(temp) for i in temp]
    load_dict['lpos'] = lpos
    
    pid = load_dict['pid']
    load_dict['ppos'] = [i/pid[-1] for i in pid]
    return load_dict
             
def loadDataAndFeature(in_file, title=False, max_len=99):
    labels = []
    documents = []
    ft_list = ['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    features = []
    with open(in_file, 'r', encoding='utf8') as f:
        for line in f:
            ft = []
            load_dict = json.loads(line)
            load_dict = getRelativePos(load_dict)
            if title:
                if ('slen' in load_dict) and ('slen' not in ft_list):
                    ft_list.append('slen')
                    
                load_dict['sents'].insert(0, load_dict['title'])
                load_dict['labels'].insert(0, 'padding')
                for k in ft_list:
                    load_dict[k].insert(0, 0)
                    
                if 'slen' in load_dict:    
                    load_dict['slen'][0] = len(load_dict['title'])
                        
                
            documents.append(load_dict['sents'][: max_len+title])
            labels.append(load_dict['labels'][: max_len+title])
                
                
            for i in load_dict['gid']:
                if i > max_len:
                    break
                ft.append([load_dict[k][i-1+title] for k in ft_list])

            features.append(ft)

    return documents, labels, features
    
def labelEncode(labels):
    en_labels = []
    for labs in labels:
        en_labs = []
        for label in labs:
            # if not label in label_map:
                # print(label)
                # continue
            en_labs.append(label_map[label])
        en_labels.append(en_labs)
    return en_labels

def encode(documents, labels, embed_map, vec_size):
        
    en_documents = []
    for sentences in documents:
        length = len(sentences)
        out_sentences = []
        for sentence in sentences:
            seq = [embed_map[w] if w in embed_map else UNKNOWN * vec_size for w in sentence]
            out_sentences.append(seq)
        en_documents.append(out_sentences)
        
    en_labels = labelEncode(labels)
    
    return en_documents, en_labels
    
def sentence_padding(en_documents, labels, n_l, vec_size, is_cutoff=True):
    pad_documents = []
    for sentences in en_documents:
        length = len(sentences)
        out_sentences = []
        for sentence in sentences:
            if len(sentence) % n_l:
                sentence = sentence + [PADDING * vec_size] * (n_l - len(sentence) % n_l)
            if is_cutoff:
                out_sentences.append(sentence[0: n_l])
            else:
                for i in range(0, len(sentence), n_l):
                    out_sentences.append(sentence[i: i+n_l])
                    # 还需要label填充
        pad_documents.append(out_sentences)
    pad_labels = labels
    return pad_documents, pad_labels
    
    
def loadEmbeddings(embed_filename):
    embed_map = {}
    with open(embed_filename, 'r', encoding='utf-8') as f:
        lenth = f.readline()
        for line in f:
            line = line[:-1]
            embed = line.split(' ')
            vec_size = len(embed[1:])
            embed_map[embed[0]] = [float(x) for x in embed[1:]]
    # embed_map['，'] = embed_map[',']
    return embed_map, vec_size
    

def getSamplesAndFeatures(in_file, embed_filename, title=False, extend_f=False):

    print('load Embeddings...')
    embed_map, vec_size = loadEmbeddings(embed_filename)
    
    documents, labels, features = loadDataAndFeature(in_file, title)

    
    en_documents, en_labels = encode(documents, labels, embed_map, vec_size)
    
    if extend_f:
        features = featuresExtend(features, documents)
    # pad_documents, pad_labels = sentence_padding(en_documents, en_labels, 30, vec_size)
    
    return en_documents, en_labels, features, vec_size 

def batchGenerator(en_documents, labels, features, batch_n, is_random=False):
    if type(labels[0][0]) in (int, str):
        mutiLabel = 0
    else:
        mutiLabel = len(labels[0])
    data = list(zip(en_documents, labels, features))
    
    data.sort(key=lambda x: len(x[0]))
    for i in range(0, len(en_documents), batch_n):
        if is_random:
            random.seed()
            mid = random.randint(0,len(en_documents)-1)
            # print(mid)
            start = max(0, mid - int(batch_n/2))
            end = min(len(en_documents), mid + math.ceil(batch_n/2))
        else:
            start = i
            end = i + batch_n
        # print(start, end)
        b_data = data[start: end]
        # b_data = data[i: i+batch_n]

        b_docs, b_labs, b_ft = zip(*b_data)
        b_ft = list(b_ft)
        b_docs = list(b_docs)
        b_labs = list(b_labs)
        max_len = len(b_docs[-1])
        if len(b_docs[0]) == max_len:
            yield b_docs, b_labs, b_ft
        else:
            sen_len = len(b_docs[0][0])
            vec_size = len(b_docs[0][0][0])
            ft_len = len(b_ft[0][0])
            
            for j in range(len(b_docs)):
                if len(b_docs[j]) < max_len:
                    l = len(b_docs[j])
                    b_docs[j] = b_docs[j] + [[PADDING * vec_size] * sen_len] * (max_len - l)
                    if not mutiLabel:
                        b_labs[j] = b_labs[j] + [LABELPAD] * (max_len - l)
                    else:
                        b_labs[j] = [b_labs[j][0] + ([LABELPAD]) * (max_len - l),
                                     b_labs[j][1] + PADDING * (max_len - l)]

                    b_ft[j] = b_ft[j] + [PADDING * ft_len] * (max_len - l)
                else:
                    break
            yield b_docs, b_labs, b_ft
          
def dataSplit(X, Y, ft=None, p=0.1):
    random.seed(312)
    test_idx = [random.randint(0,len(X)-1) for _ in range(int(len(X)*p))]
    X_test = []
    Y_test = []
    ft_test = []
    X_train = []
    Y_train = []
    ft_train = []
    for i in range(len(X)):
        if i in test_idx:
            X_test.append(X[i])
            Y_test.append(Y[i])
            if ft:
                ft_test.append(ft[i])
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])
            if ft:
                ft_train.append(ft[i])
    if not ft:
        ft_test = None
        ft_train = None
    return X_train, Y_train, ft_train, X_test, Y_test, ft_test
 
def discretePos(features):
    for feat in features:
        for f in feat:
            f[3] = math.ceil(f[0]*40)
            f[4] = math.ceil(f[1]*20)
            f[5] = math.ceil(f[2]*10)
    return features
    