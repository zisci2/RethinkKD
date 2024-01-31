#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:23:19 2023

@author: ps
"""

import argparse
import os

import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models
from torchvision import transforms as T

from trainer import train_model,get_logfile_name,print_write
from data_loader.DataLoaderCIFAR import Load_CIFAR100
from data_loader.DataLoaderImageNet import Load_ImageNet


resnets = ['resnet18', 'resnet34', 'resnet50']
students = ['resnet18', 'resnet34']
modes = ['train', 'distil'] 
data_root = {
        'CIFAR100':  '/mnt/d/data/cifar-100-python/clean_img',
        'ImageNet':'/mnt/e/dataset/ImageNet/data/ImageNet2012',
        # 'CIFAR100': 'H:\Dataset\cifar100\clean_img',
        # 'ImageNet': 'H:\Dataset\ImageNet',
        }
teacher_path = {
    "weak" : "run-teacher/CIFAR100_train_61_get-teacher-weak/checkpoint_bestAcc1.pth.tar",
    "weak_2" : "run-teacher/CIFAR100_train_61_get-teacher-weak-2/checkpoint_bestAcc1.pth.tar",
    "weak_3" : "run-teacher/CIFAR100_train_get-Tw-3/run-1-epoch45/checkpoint_bestAcc1.pth.tar",

    "strong" : "run-teacher/CIFAR100_train_80_get-teacher-strong/checkpoint_bestAcc1.pth.tar",
    "strong_2" : "run-teacher/CIFAR100_train_45_get-teacher-strong-2/checkpoint_bestAcc1.pth.tar",
    "strong_3" : "run-teacher/CIFAR100_train_get-Ts-3/run-1-epoch45/checkpoint_bestAcc1.pth.tar",

}


parser = argparse.ArgumentParser(description='Distillation from resnet50')
parser.add_argument('--mode', default='distil', choices=modes,
                    help='program mode: ' +
                        ' | '.join(resnets) +
                        ' (default: train)')
parser.add_argument('--arch', default='resnet18', choices=resnets,
                    help='model architecture: ' +
                        ' | '.join(resnets) +
                        ' (default: resnet18)')
parser.add_argument('--teacher_num', default=0, type=int,
                    help='the number of teacher models')
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch size')
parser.add_argument('--dataset', default="ImageNet_LT", # eg.CIFAR100_imb100
                    help='choose which dataset to use.')
parser.add_argument('-t', '--temp', default=10., type=float,
                    help='temperature for distillation')

parser.add_argument('--alpha', default=0.2, type=float,
                    help='weighting for hard loss during distillation')

parser.add_argument('--add_name', default="01",
                    help='Add the name of the model.')

parser.add_argument('--lr_steps', default=[0,0,0], nargs='+', type=int,
                    help="LambdaLR's step1, step2, and step3.")
parser.add_argument('--lr_CosineAnnealing', default=[0,0], nargs='+', type=int,
                    help="CosineAnnealingLR's T_max and eta_min")

parser.add_argument('--train_add_weak',default=False, action='store_true',
                    help="When training the teacher model, opt for training with a weak augmentation approach.")
parser.add_argument('--train_add_strong',default=False, action='store_true',
                    help="When training the teacher model, opt for training with a strong augmentation approach.")

parser.add_argument('--T1_add_weak',default=False, action='store_true',
                    help="When training the student model, opt for using weak augmentation training with the T1 teacher.")
parser.add_argument('--T1_add_strong',default=False, action='store_true',
                    help="When training the student model, opt for using strong augmentation with the T1 teacher.")

parser.add_argument('--T2_add_weak',default=False, action='store_true',
                    help="When training the student model, opt for using weak augmentation with the T2 teacher.")
parser.add_argument('--T2_add_strong',default=False, action='store_true',
                    help="When training the student model, opt for using strong augmentation with the T2 teacher.")


parser.add_argument('--T3_add_weak',default=False, action='store_true',
                    help="When training the student model, opt for using weak augmentation with the T3 teacher.")
parser.add_argument('--T3_add_strong',default=False, action='store_true',
                    help="When training the student model, opt for using strong augmentation with the T3 teacher.")


parser.add_argument('--S_add_weak',default=False, action='store_true',
                    help="When training the student model, choose to employ weak augmentation for training the student model.")
parser.add_argument('--S_add_strong',default=False, action='store_true',
                    help="When training the student model, choose to employ strong augmentation for training the student model.")

parser.add_argument('--next_continue',default=None,
                    help="Whether to load a checkpoint and resume training. Accepts the checkpoint file path as a parameter.")

parser.add_argument('--result', metavar='DIR',
                    help='path to results')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')



def learing_rate_scheduler(optimizer,args):

    if args.lr_steps[0] != 0:
        step1 = args.lr_steps[0]
        step2 = args.lr_steps[1]
        step3 = args.lr_steps[2]
        gamma = 0.1
        warmup_epoch = 8
        print("Scheduler step1, step2, warmup_epoch, gamma:", (step1,step2,step3, gamma))
        def lr_lambda(epoch):
            if step3 != 0 and epoch >= step3:
                lr = gamma * gamma * gamma
            elif epoch >=  step2:
                lr = gamma * gamma
            elif epoch >= step1:
                lr = gamma
            else:
                lr = 1

            """Warmup"""
            if epoch < warmup_epoch:
                lr = lr * float(1 + epoch) / warmup_epoch
            return lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif args.lr_CosineAnnealing[0] != 0:
        T_max = args.lr_CosineAnnealing[0]
        eta_min = args.lr_CosineAnnealing[1]
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        print("The learning rate scheduler is not defined, please check.")
        assert False
    return lr_scheduler



def train(args, path, log_file, device):

    if 'CIFAR100' in args.dataset:
        print("Function: load_CIFAR100.")
        data = {x: Load_CIFAR100(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                     batch_size=args.batch_size, num_workers=4,
                     shuffle=True if x == 'train' else False)
        for x in ['train', 'val', 'val_aug']} 
    elif 'ImageNet' in args.dataset:
        print("Function: Load_ImageNet.")
        data = {x: Load_ImageNet(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                    batch_size=args.batch_size, num_workers=4,
                    shuffle=True if x == 'train' else False)
            for x in ['train', 'val', 'val_aug']}    
    
    
    if "CIFAR100" in args.dataset:
        class_num = 100
    elif "CIFAR10" in args.dataset:
        class_num = 10
    elif "ImageNet" in args.dataset:
        class_num = 1000
    else:
        class_num = None
        assert False

    model_ft = models.__dict__[args.arch](weights=None)
    #model_ft = model
    if args.arch == 'resnet50' and args.mode == 'train':
        model_ft.load_state_dict(torch.load("./resnet50.pt"))
    if class_num != 1000: 
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, class_num)
    model_ft = model_ft.to(device)

    T_path = []
    if args.mode == 'distil':
        teacher = []
        if args.T1_add_weak:
            T_path.append(teacher_path['weak'])
        elif args.T1_add_strong:
            T_path.append(teacher_path['strong'])

        if args.T2_add_weak:
            T_path.append(teacher_path['weak_2'])
        elif args.T2_add_strong:
            T_path.append(teacher_path['strong_2'])

        if args.T3_add_weak:
            T_path.append(teacher_path['weak_3'])
        elif args.T3_add_strong:
            T_path.append(teacher_path['strong_3'])

        if len(T_path) != args.teacher_num:
            print("The number of loaded teachers does not match the preset. Please check.")
            assert(False)
        for i in range(args.teacher_num):

            teacher_model = models.resnet50(pretrained=False) 
            num_ftrs = teacher_model.fc.in_features
            print("class_num--->",class_num)
            teacher_model.fc = nn.Linear(num_ftrs, class_num)

            print("The loaded teacher model is:",T_path[i])
            teacher_model.load_state_dict(torch.load(T_path[i])['state_dict'])
            teacher.append(teacher_model)
    else:
        teacher = [] 

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum)
    print(args.lr_steps)
    step1 = args.lr_steps[0]
    step2 = args.lr_steps[1]
    step3 = args.lr_steps[2]
    gamma = 0.1
    warmup_epoch = 8
    exp_lr_scheduler = learing_rate_scheduler(optimizer=optimizer_ft,args=args)


    # Resume from the checkpoint.
    if args.next_continue is not None:
        checkpoint_path = args.next_continue
        if args.add_name not in checkpoint_path:
            print("The loading path for resuming from the checkpoint is incorrect. Please check.")
            assert False
        # checkpoint = torch.load(checkpoint_path + '/checkpoint_state.pth.tar')
        checkpoint = torch.load(checkpoint_path)
        model_ft.load_state_dict(checkpoint['state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer'])
        exp_lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # start_epoch = checkpoint['epoch'] + 1 
        start_epoch = checkpoint['epoch'] - 1
    else:
        start_epoch = 0
        
    writer = SummaryWriter(path)

    if args.mode == 'train':
        print_str = [f"--teacher: {f'resnet50' if teacher else 'None'}",f"--teacher number: {args.teacher_num}\n"
                    f"--dataset: {args.dataset}",f"--epochs: {args.epochs}",f"--batch size: {args.batch_size}\n"
                    f"--train_add_weak: {args.train_add_weak}", f"--train_add_strong: {args.train_add_strong}\n"]
        if step1 != 0:
            print_str.append(
                f"learing_rate_scheduler: --scheduler: LambdaLR --step1: {step1} --step2: {step2} --step3: {step3} --warmup_epoch: {warmup_epoch}\n"
            )
        elif args.lr_CosineAnnealing[0] != 0:
            print_str.append(
                f"learing_rate_scheduler: --scheduler: CosineAnnealingLR --T_max: {args.lr_CosineAnnealing[0]} --eta_min: {args.lr_CosineAnnealing[1]}"
            )

    if args.mode == 'distil':
        print_str = [f" --teacher: {f'resnet50' if teacher else 'None'}",f"--teacher number: {args.teacher_num}\n"]
        
        for i in range(args.teacher_num):
            print_str.append(
                f"--teacher_{i}: {T_path[i]}\n"
            )
                        
        print_str.extend([
                f"--dataset: {args.dataset}",f"--epochs: {args.epochs}",f"--batch size: {args.batch_size}\n",
                f"--T1_add_weak: {args.T1_add_weak}",f"--T1_add_strong: {args.T1_add_strong}\n",
                f"--T2_add_weak: {args.T2_add_weak}",f"--T2_add_strong: {args.T2_add_strong}\n",
                f"--T3_add_weak: {args.T3_add_weak}",f"--T3_add_strong: {args.T3_add_strong}\n",
                f"--S_add_weak: {args.S_add_weak}",f"--S_add_strong: {args.S_add_strong}\n",
                f"--distillation temperature: {args.temp}\n",
                f"--hard label weight: {args.alpha}",f"--soft label weight: {1. - args.alpha}\n",
        ]
                    )
        if step1 != 0:
            print_str.append(
                f"learing_rate_scheduler: --scheduler: LambdaLR --step1: {step1} --step2: {step2} --step3: {step3} --warmup_epoch: {warmup_epoch}\n"
            )
        elif args.lr_CosineAnnealing[0] != 0:
            print_str.append(
                f"learing_rate_scheduler: --scheduler: CosineAnnealingLR --T_max: {args.lr_CosineAnnealing[0]} --eta_min: {args.lr_CosineAnnealing[1]}"
            )

        # elif args.teacher_num == 1:
        #     print_str = [f"--teacher: {f'resnet50' if teacher else 'None'}",f"--teacher number: {args.teacher_num}\n"
        #                 f"--teacher_1: {T_path[0]}",f"--teacher_2: None\n"
        #                 f"--dataset: {args.dataset}",f"--epochs: {args.epochs}",f"--batch size: {args.batch_size}\n"
        #                 f"--T1_add_weak: {args.T1_add_weak}",f"--T1_add_strong: {args.T1_add_strong}\n"
        #                 f"--T2_add_weak: {args.T2_add_weak}",f"--T2_add_strong: {args.T2_add_strong}\n"
        #                 f"--S_add_weak: {args.S_add_weak}",f"--S_add_strong: {args.S_add_strong}\n"
        #                 f"--distillation temperature: {args.temp}\n"
        #                 f"--hard label weight: {args.alpha}",f"--soft label weight: {1. - args.alpha}\n"]
        #     if step1 != 0:
        #         print_str.append(
        #             f"learing_rate_scheduler: --scheduler: LambdaLR --step1: {step1} --step2: {step2} --step3: {step3} --warmup_epoch: {warmup_epoch}\n"
        #         )
        #     elif args.lr_CosineAnnealing[0] != 0:
        #         print_str.append(
        #             f"learing_rate_scheduler: --scheduler: CosineAnnealingLR --T_max: {args.lr_CosineAnnealing[0]} --eta_min: {args.lr_CosineAnnealing[1]}"
        #         )

    print_write(print_str, log_file)

    train_model(model_ft, data, optimizer_ft, 
                           exp_lr_scheduler, writer, device, args.temp, args.alpha, log_file,
                           args, start_epoch, teacher, args.epochs)


def main():
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'Mode: {args.mode}')
    if args.mode == 'train':
        print_str = [f'Training model: {args.arch}...']
        path_1 = f'runs/{args.dataset}_{args.mode}_{args.add_name}'
        os.makedirs(path_1, exist_ok=True)

        subfolders = [f for f in os.listdir(path_1) if os.path.isdir(os.path.join(path_1, f)) and "run-" in f]
        path = path_1 + f"/run-{len(subfolders)+1}-epoch{args.epochs}"    

        log_file = get_logfile_name(path=path)
        print_write(print_str, log_file)
        args.result = path
        train(args, path, log_file, device)

    elif args.mode == 'distil':
        print_str = [f'Training student: {args.arch}...']
        if args.teacher_num >= 1:
            path_1 = f'runs/{args.dataset}_{args.mode}_{args.add_name}'
        # elif args.teacher_num==2:
        #     path_1 = f'runs/{args.dataset}_{args.mode}_{args.add_name}'
        os.makedirs(path_1, exist_ok=True)

        subfolders = [f for f in os.listdir(path_1) if os.path.isdir(os.path.join(path_1, f)) and "run-" in f]
        path = path_1 + f"/run-{len(subfolders)+1}-epoch{args.epochs}"      
        
        log_file = get_logfile_name(path=path)
        print_write(print_str, log_file)

        args.result = path
        train(args, path, log_file, device)


if __name__ == '__main__':
    main()
