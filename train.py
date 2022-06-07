"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
This code is modified from Boliu.Kelvin's repository.https://github.com/awenbocc/med-vqa.
MIT License
"""
import os
import time
import torch
import utils
from datetime import datetime
import torch.nn as nn

from torch.nn.init import kaiming_uniform_, xavier_uniform_




def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp



def linear(in_dim, out_dim, bias=True):

    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()
    return lin
# Train phase
def train(args, model,question_model, train_loader,s_opt=None, s_epoch=0):

    device = args.device#cuda:0
    model = model.to(device)#BAN_Model
    question_model = question_model.to(device)#classify_model

    utils.create_dir(args.output)
    run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
    ckpt_path = os.path.join(args.output,run_timestamp)
    utils.create_dir(ckpt_path)
    logger = utils.Logger(os.path.join(ckpt_path, 'medVQA.log')).get_logger()
    logger.info(">>>The net is:")
    logger.info(model)
    logger.info(">>>The args is:")
    logger.info(args.__repr__())
    # Adamax optimizer
    optim = torch.optim.Adamax(params=model.parameters())

    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    ae_criterion = torch.nn.MSELoss()

    best_eval_score = 0
    best_epoch = 0
    # Epoch passing in training phase

    
    for epoch in range(s_epoch, args.epochs):
        total_loss = 0
        train_score = 0
        number=0
        model.train()
        # Predicting and computing score
        for i, (qid,v, q, a, anser_type, question_type, phrase_type, answer_target) in enumerate(train_loader):

            optim.zero_grad()
            if args.maml:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].to(device)
            if args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(device)
            if args.other_model:
                v = v.to(device)

            
            
            q = q.to(device)
            a = a.to(device)


            answer_target = answer_target.to(device)

            if args.autoencoder:
                last_output_close, last_output_open, a_close, a_open, decoder,q_aaatn ,qv_atten= model(v, q,a, answer_target)

            else:
                last_output_close, last_output_open, a_close, a_open = model(v, q,a, answer_target)

            preds_close, preds_open = model.classify(last_output_close, last_output_open)

            loss_an = ae_criterion(q_aaatn ,qv_atten)

            loss_close = criterion(preds_close.float(), a_close)
            loss_open = criterion(preds_open.float(),a_open)
            loss = loss_close + loss_open


            if args.autoencoder:
                loss_ae = ae_criterion(v[1], decoder)
                loss = loss + (loss_an * 1.2)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()


            total_loss += loss.item()
   
            number+= q.shape[0]


        total_loss /= len(train_loader)
 
        logger.info('-------[Epoch]:{}-------'.format(epoch))
        logger.info('[Train] Loss:{:.6f}'.format(total_loss))
        
  

   