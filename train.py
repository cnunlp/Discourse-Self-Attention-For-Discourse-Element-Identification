import utils
from model import *

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import datetime
import os
import logging

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

datastr = datetime.datetime.now().strftime('%y%m%d%H%M%S')
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='./log/sent_tag_%s.log' % datastr,
                filemode='w')

def list2tensor(x, y, ft, p_embd, device='cpu'):
    inputs = torch.tensor(x, dtype=torch.float, device=device)
    labels = torch.tensor(y, dtype=torch.long, device=device)

    tp = torch.tensor(ft, dtype=torch.float, device=device)[:, :, :6]
    return inputs, labels, tp

def getMask(ft, device='cpu'):
    slen = torch.tensor(ft, dtype=torch.long)[:, :, 6]

    s_n = slen.size(0) * slen.size(1)
    slen = slen.view(s_n)

    mask = torch.zeros((s_n, 40)) == 1
    for i, d in enumerate(slen):
        if d < 40:
            mask[i, d:] = 1
    if device == 'cuda':
        return mask.cuda()
    return mask
    
def train(model, X, Y, FT, is_gpu=False, epoch_n=10, lr=0.1, batch_n=100, title=False, is_mask=False):
    X_train, Y_train, ft_train, X_test, Y_test, ft_test = utils.dataSplit(X, Y, FT, 0.1)
    
    if(is_gpu):
        model.cuda()
        device = 'cuda'
    else:
        model.cpu()
        device = 'cpu'
        
    modelName = model.getModelName()
    if title:
        modelName += '_t'  
    logging.info(modelName)
    
    loss_function = nn.NLLLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_list = []
    acc_list = []
    last_loss = 100
    c = 0
    c1 = 0
    
    last_acc, _ = test(model, X_test, Y_test, ft_test, device, title=title, is_mask=is_mask)
    logging.info('first acc: %f' % last_acc)  
    for epoch in range(epoch_n):
        total_loss = 0
        gen = utils.batchGenerator(X_train, Y_train, ft_train, batch_n, is_random=True)
        i = 0
        model.train()
        for x, y, ft in gen:
            optimizer.zero_grad()
            
            inputs, labels, tp = list2tensor(x, y, ft, model.p_embd, device)
            
            if is_mask:
                mask = getMask(ft, device)
            else:
                mask = None

            if title:
                result = model(inputs, pos=tp, device=device, mask=mask)[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
            else:
                result = model(inputs, pos=tp, device=device, mask=mask)
            

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

        accuracy, _ = test(model, X_test, Y_test, ft_test, device, title=title, is_mask=is_mask)
        acc_list.append(accuracy)
        if last_acc < accuracy:
            last_acc = accuracy
            if accuracy > 0.58:
                torch.save(model, model_dir + '%s_%d_best.pk' % (modelName, int(epoch/20)*20))
        logging.info('%d total loss: %f accuracy: %f' % (epoch, aver_loss, accuracy))
        
        if(aver_loss > last_loss):
            c += 1
            if c == 10:
                lr = lr * 0.95
                optimizer.param_groups[0]['lr'] = lr
                logging.info('lr: %f' % lr)
                c = 0
        else:
            c = 0
            last_loss = aver_loss
        torch.save(model, model_dir + '%s_last.pk' % (modelName))

        if(lr < 0.0001) or (aver_loss < 0.5):
            break
    plt.cla()
    plt.plot(range(len(acc_list)), acc_list, range(len(loss_list)), loss_list)
    plt.legend(['acc_list', 'loss_list'])
    plt.savefig('./img/'+modelName+'.jpg')
    
def test(model, X, Y, FT, device='cpu', batch_n=1, title=False, is_mask=False):
    result_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        gen = utils.batchGenerator(X, Y, FT, batch_n)
        for x, y, ft in gen:
            
            inputs, labels, tp = list2tensor(x, y, ft, model.p_embd, device)
            
            if is_mask:
                mask = getMask(ft, device)
            else:
                mask = None
                
            if title:
                result = model(inputs, pos=tp, device=device, mask=mask)[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
            else:
                result = model(inputs, pos=tp, device=device, mask=mask)

            r_n = labels.size()[0]*labels.size()[1]
            result = result.contiguous().view(r_n, -1)
            labels = labels.view(r_n)

            result_list.append(result)
            label_list.append(labels)

    preds = torch.cat(result_list)
    labels = torch.cat(label_list)
    t_c = 0
    a = np.zeros((8, 8))
    l = preds.size()[0]
    for i in range(l):
        p = preds[i].cpu().argmax().numpy()
        r = int(labels[i].cpu().numpy())
        a[r][p] += 1
        if p == r:
            t_c += 1
    accuracy = t_c / l
    return accuracy, a   
    
def predict(model, x, ft, device='cpu', title=False):
    inputs, _, tp = list2tensor(x, [], ft, model.p_embd, device)
                
    if title:
        result = model(inputs, pos=tp, device=device)[:, 1:].contiguous()
    else:
        result = model(inputs, pos=tp, device=device)
    r_n = result.size()[0]*result.size()[1]
    result = result.contiguous().view(r_n, -1)
    return result.cpu().argmax(dim=1).tolist()

    
if __name__ == "__main__":

    in_file = './data/Ch_train.json'

    logging.info(in_file)
    
    embed_filename = './embd/new_embeddings2.txt'
    title = True
    max_len = 40
    is_topic = False
    is_suppt = False
    en_documents, en_labels, features, vec_size = utils.getSamplesAndFeatures(in_file, embed_filename, title=title)
    pad_documents, pad_labels = utils.sentence_padding(en_documents, en_labels, max_len, vec_size)
    
    is_mask = False
    
    class_n = 8
    batch_n = 50
    # is_gpu =  True
    is_gpu =  False
    
    hidden_dim = 128
    sent_dim = 128
    
    p_embd = 'add'
    p_embd_dim=16
    
    if p_embd in ['embd_b', 'embd_c', 'addv']:
        p_embd_dim = hidden_dim*2
            
    if p_embd != 'embd_c':
        features = utils.discretePos(features)

    tag_model = STWithRSbySPP(vec_size, hidden_dim, sent_dim, class_n, p_embd=p_embd, p_embd_dim=p_embd_dim, pool_type='max_pool', active_func='tanh')
    
    
    if p_embd == 'embd_b':
        tag_model.posLayer.init_embedding()
    
    model_dir = './model/roles/%s_%s/' % (tag_model.getModelName(), datastr)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    
    logging.info('start training model...')
    starttime = datetime.datetime.now()
    train(tag_model, pad_documents, pad_labels, features, is_gpu, epoch_n=700, lr=0.2, batch_n=batch_n, title=title, is_mask=is_mask)
    endtime = datetime.datetime.now()
    logging.info(endtime - starttime)

