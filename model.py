"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
This code is modified from Boliu.Kelvin's repository.https://github.com/awenbocc/med-vqa.
MIT License
"""
import torch
import torch.nn as nn
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from connect import FCNet
from connect import BCNet
from counting import Counter
from utils import tfidf_loading
from maml import SimpleCNN
from auto_encoder import Auto_Encoder_Model
from torch.nn.utils.weight_norm import weight_norm
from classify_question import typeAttention,typeAttention1,QuestionAttention1,QuestionAttention
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch.nn.functional as F

# Bilinear Attention
class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):  #128, 1024, 1024,2
        super(BiAttention, self).__init__()
        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
            name='h_mat', dim=None)
 
    def forward(self, v, q, v_mask=True):  
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)  

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, -float('inf'))

        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)

        return p.view(-1, self.glimpse, v_num, q_num), logits

class BiResNet(nn.Module):
    def __init__(self,args,dataset,priotize_using_counter=False):
        super(BiResNet,self).__init__()

        use_counter = args.use_counter if priotize_using_counter is None else priotize_using_counter
        if use_counter or priotize_using_counter:
            objects = 10 
        if use_counter or priotize_using_counter:
            counter = Counter(objects)
        else:
            counter = None
       
        b_net = []   
        q_prj = []  
        c_prj = []
        for i in range(args.glimpse):
            b_net.append(BCNet(dataset.v_dim, args.hid_dim, args.hid_dim, None, k=1))
            q_prj.append(FCNet([args.hid_dim, args.hid_dim], '', .2))
            if use_counter or priotize_using_counter:
                c_prj.append(FCNet([objects + 1, args.hid_dim], 'ReLU', .0))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.args = args

    def forward(self, v_emb, q_emb,att_p):
        b_emb = [0] * self.args.glimpse
        for g in range(self.args.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att_p[:,g,:,:])
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
        return q_emb.sum(1)


def seperate(v,q,a,att,answer_target):     

    indexs_open = []
    indexs_close = []
    for i in range(len(answer_target)):
        if answer_target[i]==0:
            indexs_close.append(i)
        else:
            indexs_open.append(i)

   
    return v[indexs_open, :, :], v[indexs_close, :, :], q[indexs_open, :, :], \
           q[indexs_close, :, :], a[indexs_open, 56:487], a[indexs_close, :56], att[indexs_open, :], att[indexs_close,
                                                                                                  :]

def linear(in_dim, out_dim, bias=True):

    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()
 
    return lin
mem_dim=1024
conv_dim=128
mlp_dim =1024

class QueryNetwork(nn.Module):
    def __init__(self, mem_dim, conv_dim):
        super(QueryNetwork, self).__init__()
        self.word_trans = linear( conv_dim, mem_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, v):
        trans_query = self.word_trans(v)  

        trans_query = torch.relu(trans_query)

        attention = torch.bmm(q, trans_query.permute(0, 2, 1)).squeeze(-1)
        attention = F.softmax(attention,dim=-1)
        ques_attn = attention.unsqueeze(-1) * q


        return ques_attn
class QueryNetwork1(nn.Module):
    def __init__(self, mem_dim, conv_dim):
        super(QueryNetwork1, self).__init__()
        self.word_trans = linear( conv_dim, mem_dim, bias=False)
        self.BatchNorm1d = nn.BatchNorm1d(12)
        self.BatchNorm1d.cuda()
    def forward(self, q, v):
        trans_query = self.word_trans(v) 

        trans_query = torch.relu(trans_query)
        attention = torch.bmm(q, trans_query.permute(0, 2, 1)).squeeze(-1)
        att=self.BatchNorm1d(attention)
        attention = F.softmax(att,dim=-1) 
        return attention


class AnswerNetwork(nn.Module):
    def __init__(self, mem_dim, conv_dim):
        super(AnswerNetwork, self).__init__()
        self.word_trans = linear( conv_dim, mem_dim, bias=False)

        self.attn = linear(mem_dim, 1)
    def forward(self, q, v,a):
        trans_query = self.word_trans(v)
        trans_query = torch.relu(trans_query)
        attentions = torch.bmm(q, trans_query.permute(0, 2, 1)).squeeze(-1)
        ques_attn = attentions * a
        ques_attn = torch.relu(ques_attn)
        attention = F.softmax(ques_attn, dim=-1)
        return attention


class BAN_Model(nn.Module):
    def __init__(self, dataset,args):
        super(BAN_Model, self).__init__()
        self.args = args
        self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.cat)
        self.q_emb = QuestionEmbedding(300, args.hid_dim, 1, False, 0.0, 'GRU')
        self.qq_emb = QueryNetwork(args.hid_dim,128)
        self.qqq_emb = QueryNetwork1(args.hid_dim, 128)
        self.ans_emb = AnswerNetwork(args.hid_dim, 128)
        self.close_att = BiAttention(dataset.v_dim, args.hid_dim, args.hid_dim, args.glimpse)
        self.close_resnet = BiResNet(args, dataset)
        self.close_classifier = SimpleClassifier(args.hid_dim, args.hid_dim * 2, dataset.num_close_candidates, args)
        self.open_att = BiAttention(dataset.v_dim, args.hid_dim, args.hid_dim, args.glimpse)
        self.open_resnet = BiResNet(args, dataset)
        self.open_classifier = SimpleClassifier(args.hid_dim, args.hid_dim * 2, dataset.num_open_candidates, args)

        self.typeatt = typeAttention(dataset.dictionary.ntoken,'./data/glove6b_init_300d.npy')
        self.qattn = QuestionAttention1(args.hid_dim)
        self.word_trans = linear(600, 300, bias=False)
        if args.maml:
            weight_path = args.data_dir + '/' + args.maml_model_path
            print('load initial weights MAML from: %s' % (weight_path))
            self.maml = SimpleCNN(weight_path, args.eps_cnn, args.momentum_cnn)

        if args.autoencoder:
            self.ae = Auto_Encoder_Model()
            weight_path = args.data_dir + '/' + args.ae_model_path
            print('load initial weights DAE from: %s' % (weight_path))
            self.ae.load_state_dict(torch.load(weight_path))
            self.convert = nn.Linear(16384, 64)

     
        if hasattr(args, 'tfidf'):

            self.w_emb = tfidf_loading(args.tfidf, self.w_emb, args)

        if args.other_model:
            pass
        

    def forward(self, v, q, a, answer_target):


        if self.args.maml:
            maml_v_emb = self.maml(v[0]).unsqueeze(1)
            v_emb = maml_v_emb 

        if self.args.autoencoder:
            encoder = self.ae.forward_pass(v[1])
            decoder = self.ae.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb

        if self.args.maml and self.args.autoencoder:
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)


        if self.args.other_model:
            pass

        type_att = self.typeatt(q)
        w_emb = self.w_emb(q)
        w_emb=self.word_trans(w_emb )
        q_emb = self.q_emb.forward_all(w_emb) 
        qa_emb = self.qq_emb.forward(q_emb, v_emb)

        q_aaatn = self.qattn.forward(w_emb, q_emb)
        qv_atten = self.qqq_emb.forward(q_emb, v_emb)
        v_open, v_close, q_open, q_close,a_open, a_close, typeatt_open, typeatt_close = seperate(v_emb,qa_emb,a,type_att,answer_target)
        att_close, _ = self.close_att(v_close,q_close)
        att_open, _ = self.open_att(v_open,q_open)

        last_output_close = self.close_resnet(v_close,q_close,att_close)
        last_output_open = self.open_resnet(v_open,q_open,att_open)


        if self.args.autoencoder:
                
                return last_output_close, last_output_open, a_close, a_open, decoder,q_aaatn , qv_atten

 
        return last_output_close, last_output_open, a_close, a_open,q_aaatn ,qv_atten

    def classify(self, close_feat, open_feat):

        return self.close_classifier(close_feat), self.open_classifier(open_feat)

    def answer_question(self,q,v,a):

        return self.ans_emb(q,v,a)

    def question_attr(self, q, v):

        return self.qqq_emb(q, v)
    
    def forward_classify(self,v,q,a,classify):

        if self.args.maml:
            maml_v_emb = self.maml(v[0]).unsqueeze(1)
            v_emb = maml_v_emb
        if self.args.autoencoder:
            encoder = self.ae.forward_pass(v[1])
            decoder = self.ae.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb
        if self.args.maml and self.args.autoencoder:
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)
        if self.args.other_model:
            pass


        type_att = self.typeatt(q)

        w_emb = self.w_emb(q)
        w_emb = self.word_trans(w_emb)
        q_emb = self.q_emb.forward_all(w_emb) 
        qa_emb = self.qq_emb.forward(q_emb, v_emb)
        answer_target = classify(q)

        _,predicted=torch.max(answer_target,1)
        v_open, v_close, q_open, q_close,a_open, a_close,typeatt_open, typeatt_close = seperate(v_emb,qa_emb,a,type_att,predicted)
        att_close, _ = self.close_att(v_close,q_close)
        att_open, _ = self.open_att(v_open,q_open)

        last_output_close = self.close_resnet(v_close,q_close,att_close)
        last_output_open = self.open_resnet(v_open,q_open,att_open)

        if self.args.autoencoder:
                return last_output_close,last_output_open,a_close,a_open, decoder
        return last_output_close,last_output_open,a_close, a_open
        


