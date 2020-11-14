import random
import math
from utils import loadDataAndFeature, dataSplit, PADDING, embd_name

LABELPAD = 0

label_map = {'padding': 0, 
             'MajorClaim': 1, 
             'Claim': 2, 
             'Premise': 3,   
             'Other': 4}
             
first_person_list = ['I', 'me', 'my', 'mine', 'myself']

forward_list = ['As a result', 'As the consequence', 'Because', 'Clearly', 'Consequently', 'Considering this subject', 'Furthermore', 'Hence', 'leading to the consequence', 'so', 'So', 'taking account on this fact', 'That is the reason why', 'The reason is that', 'Therefore', 'therefore', 'This means that', 'This shows that', 'This will result', 'Thus', 'thus', 'Thus, it is clearly seen that', 'Thus, it is seen', 'Thus, the example shows']

backward_list = ['Additionally', 'As a matter of fact', 'because', 'Besides', 'due to', 'Finally', 'First of all', 'Firstly', 'for example', 'For example', 'For instance', 'for instance', 'Furthermore', 'has proved it', 'In addition', 'In addition to this', 'In the first place', 'is due to the fact that', 'It should also be noted', 'Moreover', 'On one hand', 'On the one hand', 'On the other hand', 'One of the main reasons', 'Secondly', 'Similarly', 'since', 'Since', 'So', 'The reason', 'To begin with', 'To offer an instance', 'What is more']

thesis_list = ['All in all', 'All things considered', 'As far as I am concerned', 'Based on some reasons', 'by analyzing both the views', 'considering both the previous fact', 'Finally', 'For the reasons mentioned above', 'From explanation above', 'From this point of view', 'I agree that', 'I agree with', 'I agree with the statement that', 'I believe', 'I believe that', 'I do not agree with this statement', 'I firmly believe that', 'I highly advocate that', 'I highly recommend', 'I strongly believe that', 'I think that', 'I think the view is', 'I totally agree', 'I totally agree to this opinion', 'I would have to argue that', 'I would reaffirm my position that', 'In conclusion', 'in conclusion', 'in my opinion', 'In my opinion', 'In my personal point of view', 'in my point of view', 'In my point of view', 'In summary', 'In the light of the facts outlined above', 'it can be said that', 'it is clear that', 'it seems to me that', 'my deep conviction', 'My sentiments', 'Overall', 'Personally', 'the above explanations and example shows that', 'This, however', 'To conclude', 'To my way of thinking', 'To sum up', 'Ultimately']

rebuttal = ['Admittedly', 'although', 'Although', 'besides these advantages', 'but', 'But', 'Even though', 'even though', 'However', 'Otherwise']

indicators = [forward_list, backward_list, thesis_list, rebuttal]

def labelEncode(labels):
    en_labels = []
    for labs in labels:
        en_labs = []
        for label in labs:
            en_labs.append(label_map[label])
        en_labels.append(en_labs)
    return en_labels
    
def encodeBert(documents, labels, tokenizer, is_word=False):
    en_documents = []
    for sentences in documents:
        length = len(sentences)
        out_sentences = []
        for sentence in sentences:
            if is_word:
                seq = tokenizer.tokenize(''.join(sentence))
            else:
                seq = tokenizer.tokenize(sentence)
            seq = tokenizer.convert_tokens_to_ids(seq)
            out_sentences.append(seq)
        en_documents.append(out_sentences)
        
    en_labels = labelEncode(labels)
    
    return en_documents, en_labels
    
def getEnglishSamplesBertId(in_file, tokenizer, title=False, is_word=False):
    
    documents, labels, features = loadDataAndFeature(in_file, title)

    
    en_documents, en_labels = encodeBert(documents, labels, tokenizer, is_word)
    
    return en_documents, en_labels, features
    
def sentencePaddingId(en_documents, labels, n_l, is_cutoff=True):
    pad_documents = []
    for sentences in en_documents:
        length = len(sentences)
        out_sentences = []
        for sentence in sentences:
            if len(sentence) % n_l:
                sentence = sentence + PADDING * (n_l - len(sentence) % n_l)
            if is_cutoff:
                out_sentences.append(sentence[0: n_l])
        pad_documents.append(out_sentences)
    pad_labels = labels
    return pad_documents, pad_labels
    
def featuresExtend(features, en_documents, en_labels, tokenizer):
    en_fp_list = tokenizer.encode(' '.join(first_person_list), add_special_tokens=False)
    
    en_indicators = []
    for indi in indicators:
        en_indi = []
        for m in indi:
            en_indi.append(tokenizer.encode(m, add_special_tokens=False))
        en_indicators.append(en_indi)
        
    n_features = []
    for i, fts in enumerate(features):
        n_fts = []
        fd_c = 0
        bd_c = len(en_labels[i]) - en_labels[i].count(4) - en_labels[i].count(0)
        for j, ft in enumerate(fts):
            # first paragraph
            if ft[5] == 1:
                ft.append(1)
            else:
                ft.append(0)
            # last paragraph    
            if ft[2] == 1:
                ft.append(1)
            else:
                ft.append(0)
            # first person indictor
            fp_flag = True
            for m in en_fp_list:
                if m in en_documents[i][j]:
                    ft.append(1)
                    fp_flag = False
                    break
            if fp_flag:
                ft.append(0)
            
            # indictors
            indi_list = [0, 0, 0, 0]
            for k, indi in enumerate(en_indicators):
                for m in indi:
                    if tokenizer.decode(m) in tokenizer.decode(en_documents[i][j]):
                        indi_list[k] = 1
                        break
            ft.extend(indi_list)
            # number of preceding components
            ft.append(fd_c)
            # number of following components
            ft.append(bd_c)
            if en_labels[i][j] not in [0, 4]:
                fd_c += 1
                bd_c -= 1
            
            n_fts.append(ft)
        n_features.append(n_fts)
    return n_features


def batchGeneratorId(en_documents, labels, features, batch_n, is_random=False):
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

        b_data = data[start: end]
        
        b_docs, b_labs, b_ft = zip(*b_data)
        b_ft = list(b_ft)

        b_docs = list(b_docs)
        b_labs = list(b_labs)
        max_len = len(b_docs[-1])
        if len(b_docs[0]) == max_len:
            yield b_docs, b_labs, b_ft
        else:
            sen_len = len(b_docs[0][0])
            ft_len = len(b_ft[0][0])
            for j in range(len(b_docs)):
                if len(b_docs[j]) < max_len:
                    l = len(b_docs[j])
                    b_docs[j] = b_docs[j] + [PADDING * sen_len] * (max_len - l)
                    b_labs[j] = b_labs[j] + [LABELPAD] * (max_len - l)
                    b_ft[j] = b_ft[j] + [PADDING * ft_len] * (max_len - l)
                else:
                    break
            yield b_docs, b_labs, b_ft
