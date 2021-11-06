import datetime
import os
import logging
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='./log/sent_tag_%s.log' % datetime.datetime.now().strftime('%y%m%d%H%M%S'),
                filemode='w')
                
from transformers import BertTokenizer
from model import *
import utils_e as utils

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


import matplotlib.pyplot as plt
plt.switch_backend('Agg')


def list2tensor(x, y, ft, p_embd, device='cpu'):
    inputs = torch.tensor(x, dtype=torch.long, device=device)
    labels = torch.tensor(y, dtype=torch.long, device=device)

    tp = torch.tensor(ft, dtype=torch.float, device=device)[:, :, :6]
    return inputs, labels, tp

def train(model, X, Y, FT, is_gpu=False, epoch_n=10, lr=0.1, batch_n=100, title=False, embeddings=None, class_n=4):

    X_train, Y_train, ft_train, X_test, Y_test, ft_test = utils.dataSplit(X, Y, FT, 0.2)
    if(is_gpu):
        model.cuda()
        embeddings.cuda()
        device = 'cuda'
    else:
        model.cpu()
        embeddings.cpu()
        device = 'cpu'

    modelName = 'e_' + model.getModelName()
    if title:
        modelName += '_t' 
        
        
    logging.info(modelName)
    if class_n == 4:
        loss_function = nn.NLLLoss()
    else:
        w = torch.tensor([1., 1., 1., 1., 0.], device=device)
        loss_function = nn.NLLLoss(w)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_list = []
    acc_list = []
    last_loss = 100
    c = 0
    c1 = 0
    
    last_acc, _ = test(model, X_test, Y_test, ft_test, device, title=title, embeddings=embeddings, class_n=class_n)

    logging.info('first acc: %f' % last_acc)  

    for epoch in range(epoch_n):
        total_loss = 0
        gen = utils.batchGeneratorId(X_train, Y_train, ft_train, batch_n, is_random=True)
        i = 0
        
        for x, y, ft in gen:
            optimizer.zero_grad()
            
            inputs, labels, tp = list2tensor(x, y, ft, model.p_embd, device)  
            inputs = embeddings(inputs)

            if title:
                result = model(inputs, pos=tp, device=device)[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
            else:
                result = model(inputs, pos=tp, device=device)
            
                
            r_n = labels.size()[0]*labels.size()[1]
            result = result.contiguous().view(r_n, -1)
            labels = labels.view(r_n)
            loss = loss_function(result, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().detach().numpy()
            i += 1
            
        aver_loss = total_loss/i
        loss_list.append(aver_loss)
        
        accuracy, _ = test(model, X_test, Y_test, ft_test, device, title=title, embeddings=embeddings, class_n=class_n)
        acc_list.append(accuracy)
        if last_acc < accuracy:
            last_acc = accuracy
            if accuracy > 0.65:
                torch.save(model, model_dir + '%s_%d_best.pk' % (modelName, int(epoch/20)*20))
            c1 = 0

        logging.info('%d total loss: %f accuracy: %f' % (epoch, aver_loss, accuracy))
        
        if(aver_loss > last_loss):
            c += 1
            if c == 10:
                lr = lr * 0.95
                optimizer.param_groups[0]['lr'] = lr
                logging.info('loss lr: %f' % lr)
                c = 0
        else:
            c = 0
            last_loss = aver_loss

        torch.save(model, model_dir + '%s_last.pk' % (modelName))

        if(lr < 0.0001):
            break
    plt.cla()
    plt.plot(range(len(acc_list)), acc_list, range(len(loss_list)), loss_list)
    plt.legend(['acc_list', 'loss_list'])

    plt.savefig('./img/'+modelName+'.jpg')
    
def test(model, X, Y, FT, device='cpu', batch_n=1, title=False, embeddings=None, class_n=4):
    result_list = []
    label_list = []
    with torch.no_grad():
        gen = utils.batchGeneratorId(X, Y, FT, batch_n)
        for x, y, ft in gen:
            
            inputs, labels, tp = list2tensor(x, y, ft, model.p_embd, device)
            inputs = embeddings(inputs)
            
            if title:
                result = model(inputs, pos=tp, device=device)[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
            else:
                result = model(inputs, pos=tp, device=device)
                
            r_n = labels.size()[0]*labels.size()[1]
            result = result.contiguous().view(r_n, -1)
            labels = labels.view(r_n)
            
            result_list.append(result)
            label_list.append(labels)

    preds = torch.cat(result_list)
    labels = torch.cat(label_list)
    t_c = 0
    d = preds.size(-1)
    if class_n == 4:
        a = np.zeros((d, d))
    else:
        a = np.zeros((d-1, d-1))
    l = 0
    for i in range(preds.size(0)):
        if class_n == 4:
            p = preds[i][:].cpu().argmax().numpy()
            r = int(labels[i].cpu().numpy())
            a[r][p] += 1
            l += 1
            if p == r:
                t_c += 1
        else:
            p = preds[i][:-1].cpu().argmax().numpy()
            r = int(labels[i].cpu().numpy())
            if r != 4:
                a[r][p] += 1
                l += 1
                if p == r:
                    t_c += 1
    accuracy = t_c / l

    return accuracy, a
    
    

    
if __name__ == "__main__":

    in_file = './data/En_train.json'
    is_word=False
    class_n = 5
       

    print('load Tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('d:/project/bert-base-uncased/')

    title = True
    max_len = 40

    en_documents, en_labels, features = utils.getEnglishSamplesBertId(in_file, tokenizer, title=title, is_word=is_word)
    pad_documents, pad_labels = utils.sentencePaddingId(en_documents, en_labels, max_len)
    
    batch_n = 30
    is_gpu =  False
    
    hidden_dim = 64
    sent_dim = 64
    
    p_embd = None
    pos_dim = 0
    p_embd_dim=16
    if p_embd in ['embd_b', 'embd_c']:
        p_embd_dim = hidden_dim*2
    
    embeddings = torch.load('./embd/bert-base-uncased-word_embeddings.pkl')

    tag_model = STWithRSbySPP(embeddings.embedding_dim, hidden_dim, sent_dim, class_n, p_embd=p_embd, p_embd_dim=p_embd_dim, pool_type='max_pool', active_func='tanh')

    class_n = 4
    model_dir = './model/e_roles_%d/%s_%s/' % (class_n, tag_model.getModelName(), datetime.datetime.now().strftime('%y%m%d%H%M%S'))
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    logging.info('start training model...')
    starttime = datetime.datetime.now()
    train(tag_model, pad_documents, pad_labels, features, is_gpu, epoch_n=1500, lr=0.1, batch_n=batch_n, title=title, embeddings=embeddings, class_n=class_n)
    endtime = datetime.datetime.now()
    logging.info(endtime - starttime)

