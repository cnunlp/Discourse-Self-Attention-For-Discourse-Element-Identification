import torch
import torch.nn as nn
import torch.nn.functional as F

from subLayer import *

        
# 无句子间特征模型
class STModel(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STModel, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        
        # self.dropout = nn.Dropout(p=0.1)
        
        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)
        self.classifier = nn.Linear(self.sent_dim*2, self.class_n)
        # self.classifier2 = nn.Linear(self.sent_dim, self.class_n)
        
        self.posLayer = PositionLayer(p_embd, p_embd_dim)
        
        if p_embd == 'embd':
            self.tagLayer = nn.LSTM(self.hidden_dim*2+p_embd_dim*3, self.sent_dim, bidirectional=True)
        elif p_embd == 'cat':
            self.tagLayer = nn.LSTM(self.hidden_dim*2+3, self.sent_dim, bidirectional=True)
        else:
            self.tagLayer = nn.LSTM(self.hidden_dim*2, self.sent_dim, bidirectional=True)
        
        
        
    def init_hidden(self, batch_n, doc_l, device='cpu'):
        self.sent_hidden = (torch.rand(2, batch_n*doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01), 
                                 torch.rand(2, batch_n*doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01))
        self.tag_hidden = (torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01), 
                                torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01))
        
    def forward(self, documents, pos=None, device='cpu', mask=None):
        batch_n, doc_l, sen_l, _ = documents.size()    # documents: (batch_n, doc_l, sen_l, word_dim)
        self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)  
        documents = documents.view(batch_n*doc_l, sen_l, -1).transpose(0, 1)    # documents: (sen_l, batch_n*doc_l, word_dim)
        
        sent_out, _ = self.sentLayer(documents, self.sent_hidden)   # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
        
        if mask is None:
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))      # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) /(sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))
        
        # sentpres = torch.tanh(sent_out[-1])     # sentpres: (batch_n*doc_l, hidden_dim*2)
        
        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim*2)   # sentpres: (batch_n, doc_l, hidden_dim*2)
        # sentpres = self.dropout(sentpres)
        # self.sent1 = sentpres   
        
        sentpres = self.posLayer(sentpres, pos)
        # self.sent2 = sentpres
        
        sentpres = sentpres.transpose(0, 1)
        
        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)   # docpres: (doc_l, batch_n, output_dim*2)
        tag_out = torch.tanh(tag_out)
        # self.sent3 = tag_out
        
        tag_out = tag_out.transpose(0, 1)
        
        result = self.classifier(tag_out)

        result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
        return result
        
    def getModelName(self):
        name = 'st'
        name += '_' + str(self.hidden_dim) + '_' + str(self.sent_dim)
        if self.p_embd == 'cat':
            name += '_cp'
        elif self.p_embd =='add':
            name += '_ap'
        elif self.p_embd =='embd':
            name += '_em'
        elif self.p_embd == 'embd_a':
            name += '_em_a'
        elif self.p_embd:
            name += '_' + self.p_embd
        return name

class STWithRSbySPP(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16, pool_type='max_pool'):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPP, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.pool_type = pool_type
        
        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)
        self.classifier = nn.Linear(self.sent_dim*2 + 30, self.class_n)
        
        self.posLayer = PositionLayer(p_embd, p_embd_dim)
        self.sfLayer = InterSentenceSPPLayer(self.hidden_dim*2, pool_type = self.pool_type)
        self.rfLayer = InterSentenceSPPLayer(self.hidden_dim*2, pool_type = self.pool_type)
        
        if p_embd == 'embd':
            self.tagLayer = nn.LSTM(self.hidden_dim*2+p_embd_dim*3, self.sent_dim, bidirectional=True)
        elif p_embd == 'cat':
            self.tagLayer = nn.LSTM(self.hidden_dim*2+3, self.sent_dim, bidirectional=True)
        else:
            self.tagLayer = nn.LSTM(self.hidden_dim*2, self.sent_dim, bidirectional=True)
        
    def init_hidden(self, batch_n, doc_l, device='cpu'):
        self.sent_hidden = (torch.rand(2, batch_n*doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01), 
                                 torch.rand(2, batch_n*doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01))
        self.tag_hidden = (torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01), 
                                torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01))
        
    def forward(self, documents, pos=None, device='cpu', mask=None):
        batch_n, doc_l, sen_l, _ = documents.size()    # documents: (batch_n, doc_l, sen_l, word_dim)
        self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)  
        documents = documents.view(batch_n*doc_l, sen_l, -1).transpose(0, 1)    # documents: (sen_l, batch_n*doc_l, word_dim)
        
        sent_out, _ = self.sentLayer(documents, self.sent_hidden)   # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
        
        if mask is None:
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))      # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) /(sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))
        
        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim*2)   # sentpres: (batch_n, doc_l, hidden_dim*2)

        sentFt = self.sfLayer(sentpres)      
        
        sentpres = self.posLayer(sentpres, pos)
        sentpres = sentpres.transpose(0, 1)
        
        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)   # docpres: (doc_l, batch_n, output_dim*2)
        tag_out = torch.tanh(tag_out)
        
        tag_out = tag_out.transpose(0, 1)
        roleFt = self.rfLayer(tag_out)  
        
        tag_out = torch.cat((tag_out, sentFt, roleFt), dim=2)
        
        result = self.classifier(tag_out)

        result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
        return result
        
    def getModelName(self):
        name = 'st_rs_spp%s' % self.pool_type[0]
        name += '_' + str(self.hidden_dim) + '_' + str(self.sent_dim)
        if self.p_embd == 'cat':
            name += '_cp'
        elif self.p_embd =='add':
            name += '_ap'
        elif self.p_embd =='embd':
            name += '_em'
        elif self.p_embd == 'embd_a':
            name += '_em_a'
        elif self.p_embd:
            name += '_' + self.p_embd
        return name     


class STWithRSbySPPWithFt2(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16, ft_size=0, pool_type='max_pool'):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPPWithFt2, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.ft_size = ft_size
        self.pool_type = pool_type
        
        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)
        self.classifier = nn.Linear(self.sent_dim*2 + 30, self.class_n)
        
        self.posLayer = PositionLayer(p_embd, p_embd_dim)
        self.sfLayer = InterSentenceSPPLayer(self.hidden_dim*2, pool_type = self.pool_type)
        self.rfLayer = InterSentenceSPPLayer(self.hidden_dim*2, pool_type = self.pool_type)
        
        if p_embd == 'embd':
            self.tagLayer = nn.LSTM(self.hidden_dim*2+p_embd_dim*3+ft_size, self.sent_dim, bidirectional=True)
        elif p_embd == 'cat':
            self.tagLayer = nn.LSTM(self.hidden_dim*2+3+ft_size, self.sent_dim, bidirectional=True)
        else:
            self.tagLayer = nn.LSTM(self.hidden_dim*2+ft_size, self.sent_dim, bidirectional=True)
        
    def init_hidden(self, batch_n, doc_l, device='cpu'):
        self.sent_hidden = (torch.rand(2, batch_n*doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01), 
                                 torch.rand(2, batch_n*doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01))
        self.tag_hidden = (torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01), 
                                torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01))
        
    def forward(self, documents, pos, ft, device='cpu', mask=None):
        ft = ft[:, :, :self.ft_size]
        batch_n, doc_l, sen_l, _ = documents.size()    # documents: (batch_n, doc_l, sen_l, word_dim)
        self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)  
        documents = documents.view(batch_n*doc_l, sen_l, -1).transpose(0, 1)    # documents: (sen_l, batch_n*doc_l, word_dim)
        
        sent_out, _ = self.sentLayer(documents, self.sent_hidden)   # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
        
        if mask is None:
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))      # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) /(sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))
        
        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim*2)   # sentpres: (batch_n, doc_l, hidden_dim*2)
        sentFt = self.sfLayer(sentpres)      
        
        sentpres = self.posLayer(sentpres, pos)
        sentpres = torch.cat((sentpres, ft), dim=2)
        
        
        sentpres = sentpres.transpose(0, 1)
        
        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)   # docpres: (doc_l, batch_n, output_dim*2)
        tag_out = torch.tanh(tag_out)
        
        tag_out = tag_out.transpose(0, 1)
        roleFt = self.rfLayer(tag_out)  
        
        tag_out = torch.cat((tag_out, sentFt, roleFt), dim=2)        
        result = self.classifier(tag_out)

        result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
        return result
        
    def getModelName(self):
        name = 'st_rs_spp%s_ft2_%d' % (self.pool_type[0], self.ft_size)
        name += '_' + str(self.hidden_dim) + '_' + str(self.sent_dim) + '_' + str(self.ft_size)
        if self.p_embd == 'cat':
            name += '_cp'
        elif self.p_embd =='add':
            name += '_ap'
        elif self.p_embd =='embd':
            name += '_em'
        elif self.p_embd == 'embd_a':
            name += '_em_a'
        elif self.p_embd:
            name += '_' + self.p_embd
        return name
 