#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:58:51 2023

@author: ps
"""

import copy
import datetime
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import wasserstein_distance as w_distance
from torchvision.transforms.functional import crop
import pickle

import numpy as np
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif #mutual_info_regression
from sklearn.metrics import mutual_info_score



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)    


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def loss_kd(outputs, teacher_outputs, labels, temp, alpha):
    beta = 1. - alpha
    q = F.log_softmax(outputs/temp, dim=1)
    p = F.softmax(teacher_outputs/temp, dim=1)
    soft_loss = nn.KLDivLoss(reduction='batchmean')(q, p) * temp ** 2 
    hard_loss = nn.CrossEntropyLoss()(outputs, labels)
    KD_loss = alpha * hard_loss + beta * soft_loss 

    return KD_loss

def KL_divergence(model1_logits, model2_logits):

    probs2 = F.softmax(model2_logits, dim=1)
    log_probs1 = F.log_softmax(model1_logits, dim=1, dtype=torch.double)
    kl_div = F.kl_div(log_probs1, probs2, reduction='batchmean', log_target=False) * 10 
    return kl_div.item()


def get_logfile_name(path):
    get_time = datetime.datetime.now().strftime('%b%d_%H-%M') # 月 日 时 分
    file_name = get_time + '_log.txt'
    
    if not os.path.exists(path):  
        os.makedirs(path)  
        
    return os.path.join(path, file_name)


def print_write(print_str, log_file):
    print(*print_str)
    with open(log_file, 'a') as f:
        print(*print_str, file=f)



def get_fidelity(model1_logits, model2_logits):
    probs1 = F.softmax(model1_logits, dim=1)
    probs2 = F.softmax(model2_logits, dim=1)

    fidelity_num = torch.sum(torch.argmax(probs1, dim=1) == torch.argmax(probs2, dim=1))
    """
    print("***** debug 1")
    print(len(model1_logits)) # 128
    assert(False)
    """
    fidelity = fidelity_num.item() / len(model1_logits)
    
    return fidelity, fidelity_num

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


def get_entropy(output):
    
    probs = F.softmax(output, dim=1)
    H = entropy(probs.detach().cpu(), axis=1)
    H = np.mean(H)
    """
    print("***** debug 2")
    print(probs.cpu().size()) # torch.Size([128, 100])
    #print(probs.cpu())
    #print(H.size)
    print(H)
    assert(False)
    """
    return H


def get_mi(model1_logits, model2_logits):
    _, pred1 = model1_logits.topk(1, 1, True, True)
    #pred1 = pred1.t()
    _, pred2 = model2_logits.topk(1, 1, True, True)
    # print(pred1)
    """
    # NOT used
    probs2 = F.softmax(model2_logits, dim=1)
    mutual_info_regression()
    """
    
    mi = mutual_info_classif(pred2.cpu(), np.squeeze(pred1).cpu()) # 1st arg is feature, 2nd arg is target!
    """
    print("***** debug 3")
    print(pred1.size()) # torch.Size([128, 1])
    print(np.squeeze(pred1).size()) # torch.Size([128])
    print(mi)
    assert(False)
    """
    return mi[0]


def get_mis(model1_logits, model2_logits):
    _, pred1 = model1_logits.topk(1, 1, True, True)
    _, pred2 = model2_logits.topk(1, 1, True, True)
    
    mis = mutual_info_score(np.squeeze(pred1).cpu(), np.squeeze(pred2).cpu())
    """
    print("***** debug 4")
    print(mis)
    assert(False)
    """
    return mis


def train_teacher(train_loader, model, optimizer, scheduler, epoch, device,
          temp, alpha, args):
    
    print('=> training...')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))
    
    model.train()
    end = time.time()
    
    results_dict = {
        'T1T2_train_KL': 0.0, 
        'T1T2T3_train_KL': 0.0, 
        '1Tand1S_train_KL': 0.0,
        '2Tand1S_train_KL': 0.0,
        '3Tand1S_train_KL': 0.0,

        # fidelity
        '1Tand1S_train_f':0.0, # between the 1st T and the S
        '2Tand1S_train_f':0.0, # between the mean(2 Teachers) and the S
        '3Tand1S_train_f':0.0, # between the mean(2 Teachers) and the S
        '2Tand1T_train_f':0.0, # between the mean(2 Teachers) and the 1st T
        'T1T2_train_f': 0.0, # between the 1st T and the 2nd T
        'T1T2T3_train_f': 0.0, # between the 1st T, 2nd T and the 3th T

        # entropy
        'T1_train_entropy': 0.0, 
        'T2_train_entropy': 0.0, 
        'S_train_entropy': 0.0, 
        
        # mutual information: measuring the dependency
        '1Tand1S_train_mi':0.0, # between the 1st T and the S
        '2Tand1S_train_mi': 0, # "The mutual information between (T1_logits + T2_logits) / 2 and S."
        '3Tand1S_train_mi':0.0, 

        'T1T2_train_mi': 0.0, # between the 1st T and the 2nd T
        'T1T2T3_train_mi': 0.0, # between the 1st T and the 2nd T

        'T1SandT2S_train_mi': 0, # "The average mutual information between T1 and S, and T2 and S."

        # mutual info score: measuring the agreement
        '1Tand1S_train_mis':0.0, # between the 1st T and the S
        '2Tand1S_train_mis': 0,
        'T1T2_train_mis': 0.0, # between the 1st T and the 2nd T
        'T1SandT2S_train_mis': 0, 
        
        # record the number of batches for later averaging the above metrics!
        'batch_count': 0,

        
        }
    
    for batch_idx, (imgs_weak, imgs_strong, labels) in enumerate(train_loader):
        labels = labels.to(device)
        if args.train_add_strong:
            inputs = imgs_strong.to(device)
        if args.train_add_weak:
            inputs = imgs_weak.to(device)
            
        data_time.update(time.time() - end)
        with torch.cuda.amp.autocast():
            with torch.set_grad_enabled(True):
                outputs = model(inputs)        
                criterion = nn.CrossEntropyLoss()

                loss = criterion(outputs,labels)
                
                acc1, _ = accuracy(outputs, labels, topk=(1, 5))
                
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_time.update(time.time() - end)
                end = time.time()
                
                if batch_idx % 10 == 0:
                    progress.display(batch_idx)
        results_dict['batch_count'] += 1
    return (top1.avg, losses.avg, results_dict)


def my_train(train_loader, model, teacher, optimizer, scheduler, epoch, device,
          temp, alpha, args, stop_calculate_information):
    
    print('=> training...')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))
    
    model.train()
    end = time.time()
    
    results_dict = {
        'T1T2_train_KL': 0.0, 
        'T1T2T3_train_KL': 0.0, 
        '1Tand1S_train_KL': 0.0,
        '2Tand1S_train_KL': 0.0,
        '3Tand1S_train_KL': 0.0,

        # fidelity
        '1Tand1S_train_f':0.0, # between the 1st T and the S
        '2Tand1S_train_f':0.0, # between the mean(2 Teachers) and the S
        '3Tand1S_train_f':0.0, # between the mean(2 Teachers) and the S
        '2Tand1T_train_f':0.0, # between the mean(2 Teachers) and the 1st T
        'T1T2_train_f': 0.0, # between the 1st T and the 2nd T
        'T1T2T3_train_f': 0.0, # between the 1st T 、 2nd T and the 3th T

        # entropy
        'T1_train_entropy': 0.0, 
        'T2_train_entropy': 0.0, 
        'S_train_entropy': 0.0, 
        
        # mutual information: measuring the dependency
        '1Tand1S_train_mi':0.0, # between the 1st T and the S
        '2Tand1S_train_mi': 0, # "The mutual information between (T1_logits + T2_logits) / 2 and S."
        '3Tand1S_train_mi':0.0, 

        'T1T2_train_mi': 0.0, # between the 1st T and the 2nd T
        'T1T2T3_train_mi': 0.0, # between the 1st T and the 2nd T

        'T1SandT2S_train_mi': 0, # "The average mutual information between T1 and S, and T2 and S."

        # mutual info score: measuring the agreement
        '1Tand1S_train_mis':0.0, # between the 1st T and the S
        '2Tand1S_train_mis': 0,
        'T1T2_train_mis': 0.0, # between the 1st T and the 2nd T
        'T1SandT2S_train_mis': 0, 
        
        # record the number of batches for later averaging the above metrics!
        'batch_count': 0,
                    }
    
    for batch_idx, (imgs_weak, imgs_strong, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        imgs_weak, imgs_strong, labels = imgs_weak.to(device), imgs_strong.to(device), labels.to(device)
                
        teacher_outputs = []
        KD_loss = []
        with torch.no_grad():
            if len(teacher) > 0 :

                for teacher_num in range(len(teacher)):
                    teacher[teacher_num] = teacher[teacher_num].to(device)
                    if args.S_add_strong: # "The data types of inputs for both teacher and student are consistent."
                        teacher_outputs.append(teacher[teacher_num](imgs_strong)) 
                    elif args.S_add_weak:
                        teacher_outputs.append(teacher[teacher_num](imgs_weak)) 

    
        with torch.set_grad_enabled(True):
            if args.S_add_strong:
                outputs = model(imgs_strong) 
            elif args.S_add_weak:
                outputs = model(imgs_weak) 
            
            criterion = loss_kd
            for i in range(len(teacher)):
                KD_loss.append(criterion(outputs, teacher_outputs[i], labels, temp, alpha))
            loss = sum(KD_loss) / len(KD_loss)
            
            acc1, _ = accuracy(outputs, labels, topk=(1, 5))
            
            # losses.update(loss.item(), inputs.size(0))
            # top1.update(acc1[0], inputs.size(0))
            losses.update(loss.item(), imgs_weak.size(0))
            top1.update(acc1[0], imgs_weak.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if batch_idx % 10 == 0:
                progress.display(batch_idx)


        results_dict['batch_count'] += 1
        if stop_calculate_information:

            fedilty,fedilty_num = get_fidelity(teacher_outputs[0],outputs)
            results_dict['1Tand1S_train_f'] += fedilty
            results_dict['S_train_entropy'] += get_entropy(outputs)
            results_dict['1Tand1S_train_mi'] += get_mi(teacher_outputs[0], outputs)
            results_dict['1Tand1S_train_mis'] += get_mis(teacher_outputs[0], outputs)
            # ??? also compute the kl between T and S?
            results_dict['1Tand1S_train_KL'] += KL_divergence(teacher_outputs[0], outputs)
            

            if len(teacher) == 1:
                results_dict['T1_train_entropy'] += get_entropy(teacher_outputs[0])
                
            # if len(teacher) == 2:
            if len(teacher) >= 2:


                results_dict['T1T2_train_KL'] += KL_divergence(teacher_outputs[0], teacher_outputs[1])
                results_dict['2Tand1S_train_KL'] += KL_divergence((teacher_outputs[0]+teacher_outputs[1])/2, outputs)
                if len(teacher) == 3:
                    results_dict['T1T2T3_train_KL'] += KL_divergence((teacher_outputs[0]+teacher_outputs[1])/2,teacher_outputs[2])
                    results_dict['3Tand1S_train_KL'] += KL_divergence((teacher_outputs[0]+teacher_outputs[1]+teacher_outputs[2])/3, outputs)

                fedilty,fedilty_num = get_fidelity(teacher_outputs[0], teacher_outputs[1])
                results_dict['T1T2_train_f'] += fedilty
                fedilty,fedilty_num = get_fidelity((teacher_outputs[0]+teacher_outputs[1])/2,outputs)
                results_dict['2Tand1S_train_f'] += fedilty
                fedilty,fedilty_num = get_fidelity((teacher_outputs[0]+teacher_outputs[1])/2,teacher_outputs[0])
                results_dict['2Tand1T_train_f'] += fedilty
                if len(teacher) == 3:
                    fedilty,fedilty_num = get_fidelity((teacher_outputs[0]+teacher_outputs[1])/2,teacher_outputs[2])
                    results_dict['T1T2T3_train_f'] += fedilty
                    fedilty,fedilty_num = get_fidelity((teacher_outputs[0]+teacher_outputs[1]+teacher_outputs[2])/3,outputs)
                    results_dict['3Tand1S_train_f'] += fedilty
                
         
                results_dict['T1_train_entropy'] += get_entropy(teacher_outputs[0])
                results_dict['T2_train_entropy'] += get_entropy(teacher_outputs[1])
                
                results_dict['T1T2_train_mi'] += get_mi(teacher_outputs[0], teacher_outputs[1])
                results_dict['T1T2_train_mis'] += get_mis(teacher_outputs[0], teacher_outputs[1])
                results_dict["2Tand1S_train_mi"] += get_mi((teacher_outputs[0] + teacher_outputs[1]) / 2, outputs)
                if len(teacher) == 3:
                    results_dict['T1T2T3_train_mi'] += get_mi((teacher_outputs[0]+teacher_outputs[1])/2,teacher_outputs[2])
                    results_dict["3Tand1S_train_mi"] += get_mi((teacher_outputs[0] + teacher_outputs[1] + teacher_outputs[2]) /3, outputs)
                
                results_dict["T1SandT2S_train_mi"] += (get_mi(teacher_outputs[0], outputs) + get_mi(teacher_outputs[1], outputs)) / 2

                results_dict["2Tand1S_train_mis"] += get_mis((teacher_outputs[0] + teacher_outputs[1]) / 2, outputs)
                results_dict["T1SandT2S_train_mis"] += (get_mis(teacher_outputs[0], outputs) + get_mis(teacher_outputs[1], outputs)) / 2

                
    return (top1.avg,losses.avg, results_dict)


def my_validate(val_loader, model, teacher, device):
    
    print('=> validating...')
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')
    
    model.eval()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            end = time.time()
            
            for batch_idx, (imgs_weak,_, labels) in enumerate(val_loader):
                inputs, labels = imgs_weak.to(device), labels.to(device)

                outputs = model(inputs) 
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)
                
                acc1, _ = accuracy(outputs, labels, topk=(1, 5))
                
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % 10 == 0:
                    progress.display(batch_idx)
            
            print(' * Acc@1 {top1.avg:.3f} '
                .format(top1=top1))
            
    return top1.avg, losses.avg


# this func is to only compute metrics with augmentation!
def my_validate_aug(val_aug_loader, model, teacher, device, args, stop_calculate_information):
    
    print('=> validating...')
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_aug_loader),
        [batch_time, losses, top1],
        prefix='Test: ')
    
    results_dict = {
        'T1T2_val_KL': 0.0, 
        'T1T2T3_val_KL': 0.0, 
        '1Tand1S_val_KL': 0.0,
        '2Tand1S_val_KL': 0.0,
        '3Tand1S_val_KL': 0.0,

        # fidelity
        '1Tand1S_val_f':0.0, # between the 1st T and the S
        '2Tand1S_val_f':0.0, # between the mean(2 Teachers) and the S
        '3Tand1S_val_f':0.0, # between the mean(2 Teachers) and the S
        '2Tand1T_val_f':0.0, # between the mean(2 Teachers) and the 1st T
        'T1T2_val_f': 0.0, # between the 1st T and the 2nd T
        'T1T2T3_val_f': 0.0, # between the 1st T 、 2nd T and the 3th T

        # entropy
        'T1_val_entropy': 0.0, 
        'T2_val_entropy': 0.0, 
        'S_val_entropy': 0.0, 
        
        # mutual information: measuring the dependency
        '1Tand1S_val_mi':0.0, # between the 1st T and the S
        '2Tand1S_val_mi': 0, # "The mutual information between (T1_logits + T2_logits) / 2 and S."
        '3Tand1S_val_mi':0.0, 

        'T1T2_val_mi': 0.0, # between the 1st T and the 2nd T
        'T1T2T3_val_mi': 0.0, # between the 1st T and the 2nd T

        'T1SandT2S_val_mi': 0, # "The average mutual information between T1 and S, and T2 and S."

        # mutual info score: measuring the agreement
        '1Tand1S_val_mis':0.0, # between the 1st T and the S
        '2Tand1S_val_mis': 0,
        'T1T2_val_mis': 0.0, # between the 1st T and the 2nd T
        'T1SandT2S_val_mis': 0, 
        
        # record the number of batches for later averaging the above metrics!
        'batch_count': 0,

    }
    
    model.eval()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            end = time.time()
            
            for batch_idx, (imgs_weak, imgs_strong, labels) in enumerate(val_aug_loader):
                #data_time.update(time.time() - end)
                imgs_weak, imgs_strong, labels = imgs_weak.to(device), imgs_strong.to(device), labels.to(device)

                teacher_outputs = []
                if len(teacher) > 0 :

                    for teacher_num in range(len(teacher)):
                        teacher[teacher_num] = teacher[teacher_num].to(device)
                        if args.S_add_strong: # 
                            teacher_outputs.append(teacher[teacher_num](imgs_strong)) 
                        elif args.S_add_weak:
                            teacher_outputs.append(teacher[teacher_num](imgs_weak)) 


                #outputs = model(inputs) 
                if args.S_add_strong:
                    outputs = model(imgs_strong) 
                elif args.S_add_weak:
                    outputs = model(imgs_weak) 
                
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)
                
                acc1, _ = accuracy(outputs, labels, topk=(1, 5))
                
                #losses.update(loss.item(), inputs.size(0))
                #top1.update(acc1[0], inputs.size(0))
                losses.update(loss.item(), imgs_weak.size(0))
                top1.update(acc1[0], imgs_weak.size(0))
                
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % 10 == 0:
                    progress.display(batch_idx)


                results_dict['batch_count'] += 1
                if stop_calculate_information:

                    if len(teacher) > 0 :
                        fedilty,fedilty_num = get_fidelity(teacher_outputs[0],outputs)
                        results_dict['1Tand1S_val_f'] += fedilty
                        
                        results_dict['S_val_entropy'] += get_entropy(outputs)
                        results_dict['1Tand1S_val_mi'] += get_mi(teacher_outputs[0], outputs)
                        results_dict['1Tand1S_val_mis'] += get_mis(teacher_outputs[0], outputs)
                        # ??? also compute the kl between T and S?
                        results_dict['1Tand1S_val_KL'] += KL_divergence(teacher_outputs[0], outputs)
                        

                    if len(teacher) == 1:
                        results_dict['T1_val_entropy'] += get_entropy(teacher_outputs[0])
                        
                    # if len(teacher) == 2:
                    if len(teacher) >= 2:

                        results_dict['T1T2_val_KL'] += KL_divergence(teacher_outputs[0], teacher_outputs[1])
                        
                        if len(teacher) == 3:
                            results_dict['T1T2T3_val_KL'] += KL_divergence((teacher_outputs[0]+teacher_outputs[1])/2,teacher_outputs[2])
                            results_dict['3Tand1S_val_KL'] += KL_divergence((teacher_outputs[0]+teacher_outputs[1]+teacher_outputs[2])/3,outputs)
                        
                        fedilty,fedilty_num = get_fidelity(teacher_outputs[0], teacher_outputs[1])
                        results_dict['T1T2_val_f'] += fedilty

                        if len(teacher) == 3:
                            fedilty,fedilty_num = get_fidelity((teacher_outputs[0]+teacher_outputs[1])/2,teacher_outputs[2])
                            results_dict['T1T2T3_val_f'] += fedilty
                            fedilty,fedilty_num = get_fidelity((teacher_outputs[0]+teacher_outputs[1]+teacher_outputs[2])/3,outputs)
                            results_dict['3Tand1S_val_f'] += fedilty
                        
                        results_dict['T1_val_entropy'] += get_entropy(teacher_outputs[0])
                        results_dict['T2_val_entropy'] += get_entropy(teacher_outputs[1])
                        
                        results_dict['T1T2_val_mi'] += get_mi(teacher_outputs[0], teacher_outputs[1])
                        results_dict['T1T2_val_mis'] += get_mis(teacher_outputs[0], teacher_outputs[1])
                        
                        if len(teacher) == 3:
                            results_dict['T1T2T3_val_mi'] += get_mi((teacher_outputs[0]+teacher_outputs[1])/2,teacher_outputs[2])
                            results_dict['3Tand1S_val_mi'] += get_mi((teacher_outputs[0]+teacher_outputs[1]+teacher_outputs[2])/3,outputs)

                    # if len(teacher) == 2:
                        fedilty,fedilty_num = get_fidelity((teacher_outputs[0]+teacher_outputs[1])/2,outputs)
                        results_dict['2Tand1S_val_f'] += fedilty

                        fedilty,fedilty_num = get_fidelity((teacher_outputs[0]+teacher_outputs[1])/2,teacher_outputs[0])
                        results_dict['2Tand1T_val_f'] += fedilty
                        
                        results_dict['2Tand1S_val_KL'] += KL_divergence((teacher_outputs[0]+teacher_outputs[1])/2, outputs)

 
                        results_dict["2Tand1S_val_mi"] += get_mi((teacher_outputs[0] + teacher_outputs[1]) / 2, outputs)
                        results_dict["T1SandT2S_val_mi"] += (get_mi(teacher_outputs[0], outputs) + get_mi(teacher_outputs[1], outputs)) / 2

                        results_dict["2Tand1S_val_mis"] += get_mis((teacher_outputs[0] + teacher_outputs[1]) / 2, outputs)
                        results_dict["T1SandT2S_val_mis"] += (get_mis(teacher_outputs[0], outputs) + get_mis(teacher_outputs[1], outputs)) / 2

            print(' * Acc@1 {top1.avg:.3f} '
                .format(top1=top1))
            
    return top1.avg, losses.avg, results_dict



def train_model(model, 
                dataloaders,
                optimizer, 
                scheduler, 
                tensorboard_writer,
                device,
                temp,
                alpha,
                log_file,
                args,
                start_epoch=0,
                teacher=None,
                num_epochs=10):
    since = time.time()

    best_epoch = 0
    best_acc1 = 0.0


    stop_calculate_information = False

    history = []
    
    for epoch in range(num_epochs):
        if epoch < start_epoch:
            continue  
        
        scheduler.step()
        

        if epoch >= args.lr_steps[0] - 1:
            stop_calculate_information = True 

        if len(teacher) > 0 :
            acc1_train, loss_train, results_dict_train = my_train(dataloaders['train'], model, teacher, optimizer, scheduler, epoch, device, temp, alpha, args, stop_calculate_information)
            acc1_val, loss_val = my_validate(dataloaders['val'], model, teacher, device)
            acc1_val_aug, loss_val_aug, results_dict_val = my_validate_aug(dataloaders['val_aug'], model, teacher, device, args, stop_calculate_information)
        else:
            # below results_dicts are dummy dicts!
            acc1_train, loss_train, results_dict_train = train_teacher(dataloaders['train'], model, optimizer, scheduler, epoch, device, temp, alpha, args)
            acc1_val, loss_val, = my_validate(dataloaders['val'], model, teacher, device)
            # acc1_val_aug, loss_val_aug, results_dict_val = my_validate_aug(dataloaders['val_aug'], model, teacher, device, args)
        

        tensorboard_writer.add_scalar("train/acc",acc1_train,epoch)
        tensorboard_writer.add_scalar("train/loss",loss_train,epoch) 
        tensorboard_writer.add_scalar("val/acc",acc1_val,epoch)
        tensorboard_writer.add_scalar("val/loss",loss_val,epoch)

        if len(teacher) > 0 :
            tensorboard_writer.add_scalar("val/acc_val_aug",acc1_val_aug,epoch)
            tensorboard_writer.add_scalar("val/loss_val_aug",loss_val_aug,epoch)

        print_str = [f'{epoch+1}/{num_epochs} Acc@1: {acc1_val}']
        print_write(print_str, log_file)

        this_result = {'epoch': epoch + 1,
                       'acc1_train': acc1_train,
                       'acc1_val': acc1_val,
                       #'results_dict': results_dict
                       }
        if len(teacher) > 0 :
            this_result['results_dict_train'] = results_dict_train
            this_result['results_dict_val'] = results_dict_val
            
        history.append(this_result)

        # remember best acc@1 and save checkpoint
        is_best = acc1_val > best_acc1
        best_acc1 = max(acc1_val, best_acc1)
        if is_best:
            best_epoch = epoch + 1
            best_state = {'epoch': epoch + 1,
                          #'arch': args.arch,
                          'state_dict': model.state_dict(),
                          'acc1_train': acc1_train,
                          'best_acc1_val': best_acc1,
                          #'corresponding_acc5': acc5,
                          'optimizer' : optimizer.state_dict(),
                          'lr_scheduler':scheduler.state_dict(),
                        #   'results_dict_train': results_dict_train,
                        #   'results_dict_val': results_dict_val
                          }  
            if len(teacher) > 0:
                best_state['results_dict_train'] = results_dict_train
                best_state['results_dict_val'] = results_dict_val
            torch.save(best_state, args.result+'/checkpoint_bestAcc1.pth.tar')

            train_val_acc_gap = acc1_train - best_acc1

        
        if args.lr_steps[0] - 1 == epoch: 
            checkpoint_state = {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'lr_scheduler':scheduler.state_dict()
                          }  
            torch.save(checkpoint_state, args.result+'/checkpoint_step1.pth.tar')


        checkpoint_state = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'lr_scheduler':scheduler.state_dict()
                        }  
        torch.save(checkpoint_state, args.result+'/checkpoint_each-epoch.pth.tar')

    # save training history to pk:
    f_pkl = open(args.result+'/history.pkl', 'wb')
    pickle.dump(history,f_pkl)
    f_pkl.close()

    time_elapsed = time.time() - since  
    hours = time_elapsed // 3600  
    minutes = (time_elapsed % 3600) // 60  
    seconds = time_elapsed % 60  


    if len(teacher) > 0:
        # (1) for train:
        best_results_dict_train = best_state['results_dict_train']
        # divided the metric value by batch_count to get the avg value over batches, as the final value!
        T1T2_train_KL_value = best_results_dict_train['T1T2_train_KL'] / best_results_dict_train['batch_count'] # needed!
        T1T2T3_train_KL_value = best_results_dict_train['T1T2T3_train_KL'] / best_results_dict_train['batch_count'] # needed!
        #TKL_value = best_results_dict_train['2Tand1T_train_kl'] / best_results_dict_train['batch_count']
        _1Tand1S_train_KL_value = best_results_dict_train['1Tand1S_train_KL'] / best_results_dict_train['batch_count'] # needed!
        _2Tand1S_train_KL_value = best_results_dict_train['2Tand1S_train_KL'] / best_results_dict_train['batch_count'] # needed!
        _3Tand1S_train_KL_value = best_results_dict_train['3Tand1S_train_KL'] / best_results_dict_train['batch_count'] # needed!
        
        _1Tand1S_train_f_value = best_results_dict_train['1Tand1S_train_f'] / best_results_dict_train['batch_count'] # needed!
        _2Tand1S_train_f_value = best_results_dict_train['2Tand1S_train_f'] / best_results_dict_train['batch_count'] # needed!
        _3Tand1S_train_f_value = best_results_dict_train['3Tand1S_train_f'] / best_results_dict_train['batch_count'] # needed!
        _2Tand1T_train_f_value = best_results_dict_train['2Tand1T_train_f'] / best_results_dict_train['batch_count'] # needed!
        T1T2_train_f_value = best_results_dict_train['T1T2_train_f'] / best_results_dict_train['batch_count'] # needed!
        T1T2T3_train_f_value = best_results_dict_train['T1T2T3_train_f'] / best_results_dict_train['batch_count'] # needed!

        T1_train_en_value = best_results_dict_train['T1_train_entropy'] / best_results_dict_train['batch_count'] # needed!
        T2_train_en_value = best_results_dict_train['T2_train_entropy'] / best_results_dict_train['batch_count'] # needed!
        S_train_en_value = best_results_dict_train['S_train_entropy'] / best_results_dict_train['batch_count'] # needed!

        _1Tand1S_train_mi_value = best_results_dict_train['1Tand1S_train_mi'] / best_results_dict_train['batch_count'] # needed!
        T1T2_train_mi_value = best_results_dict_train['T1T2_train_mi'] / best_results_dict_train['batch_count'] # needed!
        T1T2T3_train_mi_value = best_results_dict_train['T1T2T3_train_mi'] / best_results_dict_train['batch_count'] # needed!

        _1Tand1S_train_mis_value = best_results_dict_train['1Tand1S_train_mis'] / best_results_dict_train['batch_count'] # needed!
        T1T2_train_mis_value = best_results_dict_train['T1T2_train_mis'] / best_results_dict_train['batch_count'] # needed!


        _2Tand1S_train_mi_value = best_results_dict_train['2Tand1S_train_mi'] / best_results_dict_train['batch_count'] 
        _3Tand1S_train_mi_value = best_results_dict_train['3Tand1S_train_mi'] / best_results_dict_train['batch_count'] 
        T1SandT2S_train_mi_value = best_results_dict_train['T1SandT2S_train_mi'] / best_results_dict_train['batch_count'] 

        _2Tand1S_train_mis_value = best_results_dict_train['2Tand1S_train_mis'] / best_results_dict_train['batch_count'] 
        T1SandT2S_train_mis_value = best_results_dict_train['T1SandT2S_train_mis'] / best_results_dict_train['batch_count']

        # (2) for val:
        best_results_dict_val = best_state['results_dict_val']
        # divided the metric value by batch_count to get the avg value over batches, as the final value!
        T1T2_val_KL_value = best_results_dict_val['T1T2_val_KL'] / best_results_dict_val['batch_count'] # needed!
        T1T2T3_val_KL_value = best_results_dict_val['T1T2T3_val_KL'] / best_results_dict_val['batch_count'] # needed!
        #TKL_value = best_results_dict_val['2Tand1T_val_kl'] / best_results_dict_val['batch_count']
        _1Tand1S_val_KL_value = best_results_dict_val['1Tand1S_val_KL'] / best_results_dict_val['batch_count'] # needed!
        _2Tand1S_val_KL_value = best_results_dict_val['2Tand1S_val_KL'] / best_results_dict_val['batch_count'] # needed!
        _3Tand1S_val_KL_value = best_results_dict_val['3Tand1S_val_KL'] / best_results_dict_val['batch_count'] # needed!
        
        _1Tand1S_val_f_value = best_results_dict_val['1Tand1S_val_f'] / best_results_dict_val['batch_count'] # needed!
        _2Tand1S_val_f_value = best_results_dict_val['2Tand1S_val_f'] / best_results_dict_val['batch_count'] # needed!
        _3Tand1S_val_f_value = best_results_dict_val['3Tand1S_val_f'] / best_results_dict_val['batch_count'] # needed!
        _2Tand1T_val_f_value = best_results_dict_val['2Tand1T_val_f'] / best_results_dict_val['batch_count'] # needed!
        T1T2_val_f_value = best_results_dict_val['T1T2_val_f'] / best_results_dict_val['batch_count'] # needed!
        T1T2T3_val_f_value = best_results_dict_val['T1T2T3_val_f'] / best_results_dict_val['batch_count'] # needed!

        T1_val_en_value = best_results_dict_val['T1_val_entropy'] / best_results_dict_val['batch_count'] # needed!
        T2_val_en_value = best_results_dict_val['T2_val_entropy'] / best_results_dict_val['batch_count'] # needed!
        S_val_en_value = best_results_dict_val['S_val_entropy'] / best_results_dict_val['batch_count'] # needed!

        _1Tand1S_val_mi_value = best_results_dict_val['1Tand1S_val_mi'] / best_results_dict_val['batch_count'] # needed!
        T1T2_val_mi_value = best_results_dict_val['T1T2_val_mi'] / best_results_dict_val['batch_count'] # needed!
        T1T2T3_val_mi_value = best_results_dict_val['T1T2T3_val_mi'] / best_results_dict_val['batch_count'] # needed!

        _1Tand1S_val_mis_value = best_results_dict_val['1Tand1S_val_mis'] / best_results_dict_val['batch_count'] # needed!
        T1T2_val_mis_value = best_results_dict_val['T1T2_val_mis'] / best_results_dict_val['batch_count'] # needed!


        _2Tand1S_val_mi_value = best_results_dict_val['2Tand1S_val_mi'] / best_results_dict_val['batch_count'] 
        _3Tand1S_val_mi_value = best_results_dict_val['3Tand1S_val_mi'] / best_results_dict_val['batch_count'] 
        T1SandT2S_val_mi_value = best_results_dict_val['T1SandT2S_val_mi'] / best_results_dict_val['batch_count'] 

        _2Tand1S_val_mis_value = best_results_dict_val['2Tand1S_val_mis'] / best_results_dict_val['batch_count'] 
        T1SandT2S_val_mis_value = best_results_dict_val['T1SandT2S_val_mis'] / best_results_dict_val['batch_count']

        print_str = [f'Training complete in {hours:.0f}h {minutes:.0f}m {seconds:.0f}s\n',  
                    f"Best validation accuracy is: {best_acc1}\n",
                    f"at epoch: {best_epoch}\n",
                    f"The accuracy gap between train and val is: {train_val_acc_gap}\n",
                    
                    #f"KL: {KL_value}\n",
                    #f"TKL: {TKL_value}\n",

                    #train
                    f"T1T2_train_KL: {T1T2_train_KL_value}\n",
                    f"1Tand1S_train_KL: {_1Tand1S_train_KL_value}\n",
                    f"2Tand1S_train_KL: {_2Tand1S_train_KL_value}\n",
                    
                    f"1Tand1S_train_f: {_1Tand1S_train_f_value}\n",
                    f"2Tand1S_train_f: {_2Tand1S_train_f_value}\n",
                    f"2Tand1T_train_f: {_2Tand1T_train_f_value}\n",
                    f"T1T2_train_f: {T1T2_train_f_value}\n",
                    
                    f"T1_train_en: {T1_train_en_value}\n",
                    f"T2_train_en: {T2_train_en_value}\n",
                    f"S_train_en: {S_train_en_value}\n",
                    
                    f"1Tand1S_train_mi: {_1Tand1S_train_mi_value}\n",
                    f"T1T2_train_mi: {T1T2_train_mi_value}\n",
                    
                    f"1Tand1S_train_mis: {_1Tand1S_train_mis_value}\n",
                    f"T1T2_train_mis: {T1T2_train_mis_value}\n",
                    
                    f"2Tand1S_train_mi: {_2Tand1S_train_mi_value}\n",
                    f"T1SandT2S_train_mi: {T1SandT2S_train_mi_value}\n",

                    f"2Tand1S_train_mis: {_2Tand1S_train_mis_value}\n",
                    f"T1SandT2S_train_mis: {T1SandT2S_train_mis_value}\n",

                    # val
                    f"T1T2_val_KL: {T1T2_val_KL_value}\n",
                    f"1Tand1S_val_KL: {_1Tand1S_val_KL_value}\n",
                    f"2Tand1S_val_KL: {_2Tand1S_val_KL_value}\n",

                    f"1Tand1S_val_f: {_1Tand1S_val_f_value}\n",
                    f"2Tand1S_val_f: {_2Tand1S_val_f_value}\n",
                    f"2Tand1T_val_f: {_2Tand1T_val_f_value}\n",
                    f"T1T2_val_f: {T1T2_val_f_value}\n",

                    f"T1_val_en: {T1_val_en_value}\n",
                    f"T2_val_en: {T2_val_en_value}\n",
                    f"S_val_en: {S_val_en_value}\n",

                    f"1Tand1S_val_mi: {_1Tand1S_val_mi_value}\n",
                    f"T1T2_val_mi: {T1T2_val_mi_value}\n",

                    f"1Tand1S_val_mis: {_1Tand1S_val_mis_value}\n",
                    f"T1T2_val_mis: {T1T2_val_mis_value}\n",

                    f"2Tand1S_val_mi: {_2Tand1S_val_mi_value}\n",
                    f"T1SandT2S_val_mi: {T1SandT2S_val_mi_value}\n",

                    f"2Tand1S_val_mis: {_2Tand1S_val_mis_value}\n",
                    f"T1SandT2S_val_mis: {T1SandT2S_val_mis_value}\n",

                    # 3T
                    f"T1T2T3_train_KL_value : {T1T2T3_train_KL_value}\n",
                    f"3Tand1S_train_KL_value : {_3Tand1S_train_KL_value}\n",
                    f"T1T2T3_train_f_value : {T1T2T3_train_f_value}\n",
                    f"3Tand1S_train_f_value : {_3Tand1S_train_f_value}\n",
                    f"T1T2T3_train_mi_value : {T1T2T3_train_mi_value}\n",
                    f"3Tand1S_train_mi_value : {_3Tand1S_train_mi_value}\n",

                    f"T1T2T3_val_KL_value : {T1T2T3_val_KL_value}\n",
                    f"3Tand1S_val_KL_value : {_3Tand1S_val_KL_value}\n",
                    f"T1T2T3_val_f_value : {T1T2T3_val_f_value}\n",
                    f"3Tand1S_val_f_value : {_3Tand1S_val_f_value}\n",
                    f"T1T2T3_val_mi_value : {T1T2T3_val_mi_value}\n",
                    f"3Tand1S_val_mi_value : {_3Tand1S_val_mi_value}\n",
                    ]
    else:
        print_str = [f'Training complete in {hours:.0f}h {minutes:.0f}m {seconds:.0f}s\n',  
                    f"Best validation accuracy is: {best_acc1}\n",
                    f"at epoch: {best_epoch}\n",
                    f"The accuracy gap between train and val is: {train_val_acc_gap}\n",
                    
                    ]
    print_write(print_str, log_file)

    return
    
