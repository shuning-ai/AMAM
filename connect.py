"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
This code is from Boliu.Kelvin's repository.https://github.com/awenbocc/med-vqa.
MIT License
"""
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

class FCNet(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()
        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if ''!=act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
          
            layers.append(nn.Dropout(dropout))#None
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if ''!=act:
            layers.append(getattr(nn, act)())#getattr用于返回一个对象属性值

        self.main = nn.Sequential(*layers)


    def forward(self, x):
        return self.main(x)


class BCNet(nn.Module):


    def __init__(self, v_dim, q_dim, h_dim, glimpse, act='ReLU', dropout=[.2, .5], k=3): 
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim;
        self.q_dim = q_dim
        self.h_dim = h_dim;
        self.glimpse = glimpse

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1]) 
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)
        if None == glimpse:
            pass
        elif glimpse <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, glimpse, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, glimpse, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, glimpse), dim=None)

    def forward(self, v, q):
        if None == self.glimpse:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_) 
            logits = d_.transpose(1, 2).transpose(2, 3)  
            return logits


        elif self.glimpse <= self.c:
            v_ = self.dropout(self.v_net(v)).unsqueeze(1)
            q_ = self.q_net(q)
            h_ = v_ * self.h_mat  
            logits = torch.matmul(h_, q_.unsqueeze(1).transpose(2, 3)) 
            logits = logits + self.h_bias
            return logits  


        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3)) 
            return logits.transpose(2, 3).transpose(1, 2)  

    def forward_with_weights(self, v, q, w):
    
        v_ = self.v_net(v).transpose(1, 2).unsqueeze(2) 
        q_ = self.q_net(q).transpose(1, 2).unsqueeze(3) 
        logits = torch.matmul(torch.matmul(v_.float(), w.unsqueeze(1).float()), q_.float()).type_as(v_) 
        logits = logits.squeeze(3).squeeze(2)

        if 1 < self.k:
            logits = logits.unsqueeze(1) 
            logits = self.p_net(logits).squeeze(1) * self.k  
         
        return logits
