import datetime
import os
import logging
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='./log/sent_tag_%s.log' % datetime.datetime.now().strftime('%y%m%d%H%M%S'),
                filemode='w')
                
from transformers import BertTokenizer
# import config
from model import STWithRSWithFt2
import utils_e as utils

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


import matplotlib.pyplot as plt
plt.switch_backend('Agg')


def list2tensor(x, y, ft, p_embd, device='cpu'):
    # print([len(j) for i in x[:3] for j in i])
    inputs = torch.tensor(x, dtype=torch.long, device=device)
    labels = torch.tensor(y, dtype=torch.long, device=device)

    tp = torch.tensor(ft, dtype=torch.float, device=device)[:, :, :6]
    
    tft = torch.tensor(ft, dtype=torch.float, device=device)[:, :, 7:]
    return inputs, labels, tp, tft

def train(model, X, Y, FT, is_gpu=False, epoch_n=10, lr=0.1, batch_n=100, title=False, embeddings=None):

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
    loss_function = nn.NLLLoss(ignore_index=4)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_list = []
    acc_list = []
    last_loss = 100
    c = 0
    c1 = 0
    
    last_acc, _ = test(model, X_test, Y_test, ft_test, device, title=title, embeddings=embeddings)
    # acc_list.append(last_acc)
    logging.info('first acc: %f' % last_acc)  
    # last_acc = max(0.6, last_acc)
    for epoch in range(epoch_n):
        total_loss = 0
        gen = utils.batchGeneratorId(X_train, Y_train, ft_train, batch_n, is_random=True)
        i = 0
        
        for x, y, ft in gen:
            optimizer.zero_grad()
            
            inputs, labels, tp, tft = list2tensor(x, y, ft, model.p_embd, device)  
            inputs = embeddings(inputs)

            if title:
                result = model(inputs, tp, tft, device=device)[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
            else:
                result = model(inputs, tp, tft, device=device)
            
                
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
        
        # if epoch % 10 == 0:
        accuracy, _ = test(model, X_test, Y_test, ft_test, device, title=title, embeddings=embeddings)
        acc_list.append(accuracy)
        if last_acc < accuracy:
            last_acc = accuracy
            logging.info('best accuracy: %f' % (accuracy))
            if accuracy > 0.77:
                torch.save(model, model_dir + '%s_%d_best.pk' % (modelName, int(epoch/10)*10))
            c1 = 0
        # else:
            # c1 += 1
            # if c1 == 20:
                # lr = lr * 0.95
                # optimizer.param_groups[0]['lr'] = lr
                # logging.info('acc lr: %f' % lr)
                # c1 = 0
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

    # plt.show()
    
def test(model, X, Y, FT, device='cpu', batch_n=1, title=False, embeddings=None):
    result_list = []
    label_list = []
    with torch.no_grad():
        gen = utils.batchGeneratorId(X, Y, FT, batch_n)
        for x, y, ft in gen:
            
            inputs, labels, tp, tft = list2tensor(x, y, ft, model.p_embd, device)
            inputs = embeddings(inputs)
            
            if title:
                result = model(inputs, tp, tft, device=device)[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
            else:
                result = model(inputs, tp, tft, device=device)
                
            r_n = labels.size()[0]*labels.size()[1]
            result = result.contiguous().view(r_n, -1)
            labels = labels.view(r_n)
            
            result_list.append(result)
            label_list.append(labels)

    preds = torch.cat(result_list)
    labels = torch.cat(label_list)
    t_c = 0
    d = preds.size(-1)
    a = np.zeros((d-1, d-1))
    # a = np.zeros((d, d))
    l = 0
    for i in range(preds.size(0)):
        p = preds[i][:-1].cpu().argmax().numpy()
        # p = preds[i][:].cpu().argmax().numpy()
        r = int(labels[i].cpu().numpy())
        if r != 4:
        # if True:
            a[r][p] += 1
            l += 1
            if p == r:
                t_c += 1
    accuracy = t_c / l

    return accuracy, a
    
    

    
if __name__ == "__main__":
    # args = config.get_args()
    # logging.info(args)
    # main(args)
    

    in_file = './data/AAEtrain3.json'
    is_word=False
    class_n = 5
       

    print('load Tokenizer...')
    # tokenizer = BertTokenizer.from_pretrained('D:/project/bert-base-uncased/')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    title = True
    max_len = 40

    en_documents, en_labels, features = utils.getEnglishSamplesBertId(in_file, tokenizer, title=title, is_word=is_word)
    pad_documents, pad_labels = utils.sentencePaddingId(en_documents, en_labels, max_len)
    
    n_features = utils.featuresExtend(features, en_documents, en_labels, tokenizer)
    ft_size = len(n_features[0][0])-7 
    
    batch_n = 30
    is_gpu =  True
    
    hidden_dim = 64
    sent_dim = 64
    
    p_embd = 'add'
    pos_dim = 0
    p_embd_dim=16
    if p_embd in ['embd_b', 'embd_c']:
        p_embd_dim = hidden_dim*2
    
    embeddings = torch.load('./embd/bert-base-uncased-word_embeddings.pkl')
    # embeddings.weight.requires_grad = True
    
    # tag_model = torch.load('./model/STE_model_128_128_last.pk') 
    model_dir = './model/e_roles_3/%s/' % datetime.datetime.now().strftime('%y%m%d%H%M%S')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    tag_model = STWithRSWithFt2(embeddings.embedding_dim, hidden_dim, sent_dim, class_n, p_embd=p_embd, p_embd_dim=p_embd_dim, ft_size=ft_size)



    
    
    logging.info('start training model...')
    starttime = datetime.datetime.now()
    train(tag_model, pad_documents, pad_labels, n_features, is_gpu, epoch_n=1500, lr=0.1, batch_n=batch_n, title=title, embeddings=embeddings)
    endtime = datetime.datetime.now()
    logging.info(endtime - starttime)

