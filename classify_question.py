"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
This code is modified by shuning from Boliu.Kelvin's repository.https://github.com/awenbocc/med-vqa.
MIT License
"""
import torch
import torch.nn as nn
from language_model import WordEmbedding,QuestionEmbedding
import argparse
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch.nn.functional as F



def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def linear(in_dim, out_dim, bias=True):

    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()
 
    return lin


class QuestionAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.tanh_gate = linear(300 + dim, dim)
        self.sigmoid_gate = linear(300 + dim, dim)
        self.attn = linear(dim, 1)
        self.dim = dim
        self.sofmax=nn.Softmax(dim=1)


    def forward(self, context, question):
        concated = torch.cat([context, question], -1)  
 
        concated = torch.mul(torch.tanh(self.tanh_gate(concated)), torch.sigmoid(self.sigmoid_gate(concated)))  #b*12*1024 64*12*1024l论文公式4d
   

        a = self.attn(concated) 

        attn = F.softmax(a.squeeze(), 1) 


        ques_attn = torch.bmm(attn.unsqueeze(1), question).squeeze()

        return ques_attn


class QuestionAttention1(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.tanh_gate = linear(300 + dim, dim)
        self.sigmoid_gate = linear(300 + dim, dim)
        self.attn = linear(dim, 1)
        self.dim = dim
        self.mm=nn.BatchNorm1d(12)
        self.mm.cuda()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, context, question):  
        concated = torch.cat([context, question], -1)  
        concated = torch.mul(torch.tanh(self.tanh_gate(concated)), torch.sigmoid(self.sigmoid_gate(concated)))  

        a = self.attn(concated) 

        m = self.mm(a)

        attn = F.softmax(m.squeeze(), 1) 



        return attn



class typeAttention(nn.Module):
    def __init__(self, size_question, path_init):
        super(typeAttention, self).__init__()
        self.w_emb = WordEmbedding(size_question, 300, 0.0, False)
        self.w_emb.init_embedding(path_init)
        self.q_emb = QuestionEmbedding(300, 1024, 1, False, 0.0, 'GRU')
        self.q_final = QuestionAttention(1024)
        self.f_fc1 = linear(1024, 2048)
        self.f_fc2 = linear(2048, 1024)
        self.f_fc3 = linear(1024, 1024)


    def forward(self, question):
        w_emb = self.w_emb(question)

        q_emb = self.q_emb.forward_all(w_emb)  

        q_final = self.q_final.forward(w_emb, q_emb)  
        x_f = self.f_fc1(q_final)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.dropout(x_f)
        x_f = F.relu(x_f)
        x_f = self.f_fc3(x_f)

        return x_f


class typeAttention1(nn.Module):
    def __init__(self, size_question, path_init):
        super(typeAttention1, self).__init__()
        self.w_emb = WordEmbedding(size_question, 300, 0.0, False)
        self.w_emb.init_embedding(path_init)  # None
        self.q_emb = QuestionEmbedding(300, 1024, 1, False, 0.0, 'GRU')
        self.q_final = QuestionAttention(1024)
        self.f_fc1 = linear(1024, 2048)
        self.f_fc2 = linear(2048, 1024)
        self.f_fc3 = linear(1024, 1024)  


    def forward(self, question):
        w_emb = self.w_emb(question)
        q_emb = self.q_emb.forward_all(w_emb) 
        q_final = self.q_final.forward_all(w_emb, q_emb)  

        return q_final


class classify_model(nn.Module): 
    def __init__(self,size_question,path_init):
        super(classify_model,self).__init__()
        self.w_emb = WordEmbedding(size_question,300, 0.0, False)
        self.w_emb.init_embedding(path_init)
        self.q_emb = QuestionEmbedding(300, 1024 , 1, False, 0.0, 'GRU')
        self.q_final = QuestionAttention(1024)
        self.f_fc1 = linear(1024,256)
        self.f_fc2 = linear(256,64)
        self.f_fc3 = linear(64,2)



    def forward(self,question):

        w_emb = self.w_emb(question)
        q_emb = self.q_emb.forward_all(w_emb)  
        q_final = self.q_final(w_emb,q_emb) 

        x_f = self.f_fc1(q_final)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.dropout(x_f)
        x_f = F.relu(x_f)
        x_f = self.f_fc3(x_f)

        return x_f


def parse_args():
    parser = argparse.ArgumentParser(description="Med VQA over MAC")
    # GPU config
    parser.add_argument('--seed', type=int, default=88
                        , help='random seed for gpu.default:5')
    parser.add_argument('--gpu', type=int, default=1,
                        help='use gpu device. default:0')
    args = parser.parse_args()
    return args





        
        









