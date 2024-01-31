import argparse
import os
import time

import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models
from torchvision import transforms as T

# from trainer_diversity_affinity import train_model,get_logfile_name,print_write

from DataLoaderCIFAR_diversity_affinity import Load_CIFAR100
from DataLoaderImageNet_diversity_affinity import Load_ImageNet


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


def my_teacher(val_loader, teacher, device,
                use_strong = False,
                use_weak = False,
                use_base = False,):
    
    print('=> validating...')
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')
    
    teacher.eval()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            end = time.time()
            
            # for batch_idx, (imgs_weak,_, labels) in enumerate(val_loader):
            for batch_idx, (imgs_weak, imgs_strong, imgs_base, labels) in enumerate(val_loader):
                # inputs, labels = imgs_weak.to(device), labels.to(device) # 
                # if use_base:
                #     inputs, labels = imgs_base.to(device), labels.to(device) # 
                # else:
                #     inputs, labels = imgs_weak.to(device), labels.to(device) # 
                if sum([use_weak, use_strong]) == 1: 
                    if use_weak:
                        inputs, labels = imgs_weak.to(device), labels.to(device)
                    elif use_strong:
                        inputs, labels = imgs_strong.to(device), labels.to(device)
                    else:
                        assert False
                else:
                    if use_base:
                        inputs, labels = imgs_base.to(device), labels.to(device)
                    elif not use_base:
                        inputs, labels = imgs_weak.to(device), labels.to(device)
                    else:
                        assert False
                


                outputs = teacher(inputs) 
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



def my_train(train_loader, model, teacher, device,
          temp, alpha,
          use_strong = False,
          use_weak = False,
          use_base = False,
          ):
    
    print('=> training...')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        # prefix="Epoch: [{}]".format(epoch)
        )
    
    # model.train()
    model.eval()
    end = time.time()
    
#    img_weak, img_strong,img_base, label
    for batch_idx, (imgs_weak, imgs_strong, imgs_base, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        imgs_weak, imgs_strong,imgs_base, labels = imgs_weak.to(device), imgs_strong.to(device),imgs_base.to(device), labels.to(device)
        
        teacher_outputs = []
        KD_loss = []
        with torch.no_grad():
            if len(teacher) > 0 :

                for teacher_num in range(len(teacher)):
                #     teacher[teacher_num] = teacher[teacher_num].to(device)
                #     if S_add_strong: 
                #         teacher_outputs.append(teacher[teacher_num](imgs_strong)) 
                #     elif S_add_weak:
                #         teacher_outputs.append(teacher[teacher_num](imgs_weak)) 
                    if use_weak:
                        teacher_outputs.append(teacher[teacher_num](imgs_weak)) 
                    if use_strong:
                        teacher_outputs.append(teacher[teacher_num](imgs_strong)) 
                    if use_base:
                        teacher_outputs.append(teacher[teacher_num](imgs_base)) 
    
        # with torch.set_grad_enabled(True):
        # with torch.set_grad_enabled(False):
            # if S_add_strong:
            #     outputs = model(imgs_strong) 
            # elif S_add_weak:
            #     outputs = model(imgs_weak) 

            if use_weak:
                outputs = model(imgs_weak) 
            if use_strong:
                outputs = model(imgs_strong)  
            if use_base:
                outputs = model(imgs_base) 

            criterion = loss_kd
            for i in range(len(teacher)):
                KD_loss.append(criterion(outputs, teacher_outputs[i], labels, temp, alpha))
            loss = sum(KD_loss) / len(KD_loss)
            
            acc1, _ = accuracy(outputs, labels, topk=(1, 5))
            
            # losses.update(loss.item(), inputs.size(0))
            # top1.update(acc1[0], inputs.size(0))
            losses.update(loss.item(), imgs_weak.size(0))
            top1.update(acc1[0], imgs_weak.size(0))
            
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if batch_idx % 10 == 0:
                progress.display(batch_idx)
    print(' * Acc@1 {top1.avg:.3f} '
        .format(top1=top1))
    return top1.avg,losses.avg


def my_validate(val_loader, model, teacher, device,use_base):
    
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
            
            # for batch_idx, (imgs_weak,_, labels) in enumerate(val_loader):
            for batch_idx, (imgs_weak, imgs_strong, imgs_base, labels) in enumerate(val_loader):
                # inputs, labels = imgs_weak.to(device), labels.to(device) # 
                if use_base:
                    inputs, labels = imgs_base.to(device), labels.to(device) # 
                else:
                    inputs, labels = imgs_weak.to(device), labels.to(device) # 

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
def my_validate_aug(val_aug_loader, model, teacher, device, use_strong=True):
    
    print('=> validating...')
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_aug_loader),
        [batch_time, losses, top1],
        prefix='Test: ')
    

    
    model.eval()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            end = time.time()
            
            # for batch_idx, (imgs_weak, imgs_strong, labels) in enumerate(val_aug_loader):
            for batch_idx, (imgs_weak, imgs_strong,imgs_base, labels) in enumerate(val_aug_loader):
                #data_time.update(time.time() - end)
                imgs_weak, imgs_strong, labels = imgs_weak.to(device), imgs_strong.to(device), labels.to(device)

                #outputs = model(inputs) 
                # if S_add_strong:
                #     outputs = model(imgs_strong) 
                # elif S_add_weak:
                #     outputs = model(imgs_weak) 
                if use_strong:
                    outputs = model(imgs_strong) 
                else:
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

            print(' * Acc@1 {top1.avg:.3f} '
                .format(top1=top1))
            
    return top1.avg, losses.avg

def get_logfile_name(path,student_name):
    # get_time = datetime.datetime.now().strftime('%b%d_%H-%M') # 月 日 时 分
    file_name = student_name + '_log.txt'
    
    if not os.path.exists(path):  
        os.makedirs(path)  
        
    return os.path.join(path, file_name)


def print_write(print_str, log_file):
    print(*print_str)
    with open(log_file, 'a') as f:
        print(*print_str, file=f)

data_path = {
        'CIFAR100':  '/mnt/d/data/cifar-100-python/clean_img',
        'CIFAR100_imb100':  '/mnt/d/data/cifar-100-python/clean_img',
        'CIFAR10': '/mnt/d/data/cifar-10-batches-py/clean_img',
        'ImageNet':'/mnt/e/dataset/ImageNet/data/ImageNet2012',
        'ImageNet_LT':'/mnt/e/dataset/ImageNet/data/ImageNet2012',
        }


students_path = {
    # cifar100_imb100
    # "T1s" : "run-teacher/CIFAR100_train_80_get-teacher-strong/checkpoint_bestAcc1.pth.tar",
    # "T1w" : "run-teacher/CIFAR100_train_61_get-teacher-weak/checkpoint_bestAcc1.pth.tar",
    # "T2s" : "run-teacher/CIFAR100_train_45_get-teacher-strong-2/checkpoint_bestAcc1.pth.tar",
    # "T2w" : "run-teacher/CIFAR100_train_61_get-teacher-weak-2/checkpoint_bestAcc1.pth.tar",

    # # "only_Sw" : "暂时没用/11111-runs-save/CIFAR100_imb100_train_90_only_Sw/checkpoint_bestAcc1.pth.tar",
    # # "only_Ss" : "暂时没用/11111-runs-save/CIFAR100_imb100_train_75_only_Ss/checkpoint_bestAcc1.pth.tar",
    # "T1w_Sw" : "epoch_175-02/CIFAR100_imb100_distil_175_T1w_Sw/checkpoint_bestAcc1.pth.tar",
    # "T1w_Ss" : "epoch_175-02/CIFAR100_imb100_distil_175_T1w_Ss/checkpoint_bestAcc1.pth.tar",
    # "T1s_Sw" : "epoch_175-02/CIFAR100_imb100_distil_175_T1s_Sw/checkpoint_bestAcc1.pth.tar",
    # "T1s_Ss" : "epoch_175-02/CIFAR100_imb100_distil_175_T1s_Ss/checkpoint_bestAcc1.pth.tar",

    # "T1w_T2w_Sw" : "epoch_175/CIFAR100_imb100_distil_T1w_T2w_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2w_Ss" : "epoch_175/CIFAR100_imb100_distil_T1w_T2w_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_Sw" : "epoch_175/CIFAR100_imb100_distil_T1s_T2w_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_Ss" : "epoch_175/CIFAR100_imb100_distil_T1s_T2w_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_Sw" : "epoch_175/CIFAR100_imb100_distil_T1w_T2s_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_Ss" : "epoch_175/CIFAR100_imb100_distil_T1w_T2s_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_Sw" : "epoch_175/CIFAR100_imb100_distil_T1s_T2s_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_Ss" : "epoch_175/CIFAR100_imb100_distil_T1s_T2s_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",

    # # cifar100
    "T1s" : "run-teacher/CIFAR100_train_80_get-teacher-strong/checkpoint_bestAcc1.pth.tar",
    "T1w" : "run-teacher/CIFAR100_train_61_get-teacher-weak/checkpoint_bestAcc1.pth.tar",
    "T2s" : "run-teacher/CIFAR100_train_45_get-teacher-strong-2/checkpoint_bestAcc1.pth.tar",
    "T2w" : "run-teacher/CIFAR100_train_61_get-teacher-weak-2/checkpoint_bestAcc1.pth.tar",
    
    "T1w_Sw" : "run-cifar100/CIFAR100_distil_T1w_Sw/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1s_Ss" : "run-cifar100/CIFAR100_distil_T1s_Ss/run-3-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1w_Ss" : "run-cifar100/CIFAR100_distil_T1w_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1s_Sw" : "run-cifar100/CIFAR100_distil_T1s_Sw/run-2-epoch200/checkpoint_bestAcc1.pth.tar",

    "T1w_T2w_Sw":"run-cifar100/CIFAR100_distil_T1w_T2w_Sw/run-4-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1w_T2w_Ss":"run-cifar100/CIFAR100_distil_T1w_T2w_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1s_T2w_Sw":"run-cifar100/CIFAR100_distil_T1s_T2w_Sw/run-4-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1s_T2w_Ss":"run-cifar100/CIFAR100_distil_T1s_T2w_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1w_T2s_Sw":"run-cifar100/CIFAR100_distil_T1w_T2s_Sw/run-3-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1w_T2s_Ss":"run-cifar100/CIFAR100_distil_T1w_T2s_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1s_T2s_Sw":"run-cifar100/CIFAR100_distil_T1s_T2s_Sw/run-3-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1s_T2s_Ss":"run-cifar100/CIFAR100_distil_T1s_T2s_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar"


    # ImageNet_LT
    # "T1w" : "run-teacher/T1w_ImageNetLT-epoch60/run-1-epoch60/checkpoint_bestAcc1.pth.tar",
    # "T2w" : "run-teacher/T2w_ImageNetLT-epoch60/run-2-epoch60/checkpoint_bestAcc1.pth.tar",
    # "T1s" : "run-teacher/T1s_ImageNetLT-epoch60/run-1-epoch60/checkpoint_bestAcc1.pth.tar",
    # "T2s" : "run-teacher/T2s_ImageNetLT-epoch60/run-2-epoch60/checkpoint_bestAcc1.pth.tar",

    # "T1w_T2w_Sw" : "ImageNet_LT/ImageNet_LT_distil_T1w_T2w_Sw/run-1-epoch165/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2w_Ss" : "ImageNet_LT/ImageNet_LT_distil_T1w_T2w_Ss/run-1-epoch165/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_Sw" : "ImageNet_LT/ImageNet_LT_distil_T1s_T2w_Sw/run-1-epoch165/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_Ss" : "ImageNet_LT/ImageNet_LT_distil_T1s_T2w_Ss/run-1-epoch165/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_Sw" : "ImageNet_LT/ImageNet_LT_distil_T1w_T2s_Sw/run-1-epoch165/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_Ss" : "ImageNet_LT/ImageNet_LT_distil_T1w_T2s_Ss/run-1-epoch165/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_Sw" : "ImageNet_LT/ImageNet_LT_distil_T1s_T2s_Sw/run-1-epoch165/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_Ss" : "ImageNet_LT/ImageNet_LT_distil_T1s_T2s_Ss/run-1-epoch165/checkpoint_bestAcc1.pth.tar",

    # # ImageNet balance 
    # "T1w" : "run-teacher/ImageNet_balance-T1w/run-9-epoch30/checkpoint_bestAcc1.pth.tar",
    # "T2w" : "run-teacher/ImageNet_balance-T2w/run-10-epoch30/checkpoint_bestAcc1.pth.tar",
    # "T1s" : "run-teacher/ImageNet_balance-T1s/run-11-epoch30/checkpoint_bestAcc1.pth.tar",
    # "T2s" : "run-teacher/ImageNet_balance-T2s/run-12-epoch30/checkpoint_bestAcc1.pth.tar",

    # "T1w_T2w_Sw" : "ImageNet/ImageNet_distil_T1w_T2w_Sw/run-6-epoch60/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2w_Ss" : "ImageNet/ImageNet_distil_T1w_T2w_Ss/run-6-epoch60/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_Sw" : "ImageNet/ImageNet_distil_T1s_T2w_Sw/run-3-epoch60/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_Ss" : "ImageNet/ImageNet_distil_T1s_T2w_Ss/run-2-epoch60/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_Sw" : "ImageNet/ImageNet_distil_T1w_T2s_Sw/run-1-epoch60/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_Ss" : "ImageNet/ImageNet_distil_T1w_T2s_Ss/run-2-epoch60/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_Sw" : "ImageNet/ImageNet_distil_T1s_T2s_Sw/run-2-epoch60/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_Ss" : "ImageNet/ImageNet_distil_T1s_T2s_Ss/run-2-epoch60/checkpoint_bestAcc1.pth.tar",

}


teacher_path = {
    # "resnet50.pt",
    # # cifar100 / cifar100_ibm100
    "weak" : r"run-teacher/CIFAR100_train_61_get-teacher-weak/checkpoint_bestAcc1.pth.tar",
    "weak_2" : r"run-teacher/CIFAR100_train_61_get-teacher-weak-2/checkpoint_bestAcc1.pth.tar",
    "strong" : r"run-teacher/CIFAR100_train_80_get-teacher-strong/checkpoint_bestAcc1.pth.tar",
    "strong_2" : r"run-teacher/CIFAR100_train_45_get-teacher-strong-2/checkpoint_bestAcc1.pth.tar",
 
    # # ImageNet_LT
    # "weak" : r"run-teacher/T1w_ImageNetLT-epoch60/run-1-epoch60/checkpoint_bestAcc1.pth.tar",
    # "weak_2" : r"run-teacher/T2w_ImageNetLT-epoch60/run-2-epoch60/checkpoint_bestAcc1.pth.tar",
    # "strong" : r"run-teacher/T1s_ImageNetLT-epoch60/run-1-epoch60/checkpoint_bestAcc1.pth.tar",
    # "strong_2" : r"run-teacher/T2s_ImageNetLT-epoch60/run-2-epoch60/checkpoint_bestAcc1.pth.tar",

    # # ImageNet balance
    # "weak" : "run-teacher/ImageNet_balance-T1w/run-9-epoch30/checkpoint_bestAcc1.pth.tar",
    # "weak_2" : "run-teacher/ImageNet_balance-T2w/run-10-epoch30/checkpoint_bestAcc1.pth.tar",
    # "strong" : "run-teacher/ImageNet_balance-T1s/run-11-epoch30/checkpoint_bestAcc1.pth.tar",
    # "strong_2" : "run-teacher/ImageNet_balance-T2s/run-12-epoch30/checkpoint_bestAcc1.pth.tar",
}

if __name__ == "__main__":
    # dataset = "CIFAR100_imb100"
    dataset = "CIFAR100"
    # dataset = "ImageNet_LT"
    # dataset = "ImageNet"
    # arch = "resnet18" 
    mode = "distil"
    device = 'cuda:0'
    data_root = data_path[dataset]
    # teacher_num = 2
    temp = 3.0 
    alpha = 0.6
    

    for student_name in students_path:
        if dataset not in students_path[student_name]:
            print(dataset)
            print(students_path[student_name])
            if "teacher" not in students_path[student_name]:
                assert False
            if dataset == 'CIFAR100' and "CIFAR100_imb100" in students_path[student_name]:
                print(dataset)
                print(students_path[student_name])
                assert False
            if dataset == 'ImageNet' and "ImageNet_LT" in students_path[student_name]:
                print(dataset)
                print(students_path[student_name])
                assert False
        # student_name = 'T1w_T2w_Sw'
        T1_add_weak = True if 'T1w' in student_name else False
        T1_add_strong = True if 'T1s' in student_name else False
        T2_add_weak = True if 'T2w' in student_name else False
        T2_add_strong = True if 'T2s' in student_name else False

        only_student = False
        only_teacher = False
        one_teacher = False
        two_teacher = False
        if 'only' in student_name:
            arch = "resnet18"
            teacher_num = 0
            only_student = True
        elif "Ss" not in student_name and "Sw" not in student_name: 
            arch = "resnet50"
            teacher_num = 1 
            only_teacher = True
        elif sum([T1_add_weak,T1_add_strong,T2_add_weak,T2_add_strong]) == 1:
            arch = "resnet18"
            teacher_num = 1
            one_teacher = True
        elif sum([T1_add_weak,T1_add_strong,T2_add_weak,T2_add_strong]) == 2:
            arch = "resnet18"
            teacher_num = 2
            two_teacher = True
        else:
            assert False


        file_name = get_logfile_name("calculate_diversity_affinity/",student_name)


        if "bestAcc1" not in students_path[student_name]:
            assert False


        # if "CIFAR100" in dataset:
        #     class_num = 100
        if 'CIFAR100' in dataset:
            dataloaders = {x: Load_CIFAR100(data_root=data_root, dataset=dataset, phase=x,
                            batch_size=128, num_workers=4,
                            shuffle=False)
            for x in ['train', 'val','val_aug']} 
            class_num = 100
        elif 'ImageNet' in dataset:
            dataloaders = {x: Load_ImageNet(data_root=data_root, dataset=dataset, phase=x,
                            batch_size=128, num_workers=4,
                            shuffle=False)
            for x in ['train', 'val','val_aug']} 
            class_num = 1000

        model_ft = models.__dict__[arch](weights=None)
        #model_ft = model
        # if arch == 'resnet50' and mode == 'train':
        # 
        #     model_ft.load_state_dict(torch.load("./resnet50.pt"))
        if class_num != 1000: #
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, class_num)
        model_ft.load_state_dict(torch.load(students_path[student_name])['state_dict'])
        print("The model parameters have been successfully loaded.")
        model_ft = model_ft.to(device)

        T_path = []
        if mode == 'distil':
            teacher = []

            if T1_add_weak:
                T_path.append(teacher_path['weak'])
            elif T1_add_strong:
                T_path.append(teacher_path['strong'])

            if T2_add_weak:
                T_path.append(teacher_path['weak_2'])
            elif T2_add_strong:
                T_path.append(teacher_path['strong_2'])

            if len(T_path) != teacher_num:
                print("The number of loaded teachers does not match the preset. Please check.")
                print(len(T_path))
                print(teacher_num)
                assert(False)

            for i in range(teacher_num):

                teacher_model = models.resnet50(pretrained=False) 
                num_ftrs = teacher_model.fc.in_features
                print("class_num--->",class_num)
                teacher_model.fc = nn.Linear(num_ftrs, class_num)

                print("The loaded teacher model is:",T_path[i])
                teacher_model.load_state_dict(torch.load(T_path[i])['state_dict'])
                # teacher_model.load_state_dict(torch.load("./resnet50.pt")) # 测试imagenet_LT数据的训练所加载的预训练教师模型
                teacher.append(teacher_model.to(device))
        else:
            teacher = []

        # if 'CIFAR100' in dataset:
        #     
        #     dataloaders = {x: Load_CIFAR100(data_root=data_root, dataset=dataset, phase=x,
        #                     batch_size=128, num_workers=4,
        #                     shuffle=True if x == 'train' else False)
        #     for x in ['train', 'val', 'val_aug']} 
        if only_teacher or only_student:
            # acc1_train_weak, loss_train_weak = my_teacher(val_loader=dataloaders['train'], teacher=model_ft, device=device, use_weak=True)
            # acc1_train_strong, loss_train_strong = my_teacher(val_loader=dataloaders['train'], teacher=model_ft, device=device, use_strong=True)
            # acc1_train_base, loss_train_base = my_teacher(val_loader=dataloaders['train'], teacher=model_ft, device=device, use_base=True)
            
            # acc1_val_weak, loss_val_weak = my_teacher(val_loader=dataloaders['val'],  teacher=model_ft, device=device, use_weak=True)
            # acc1_val_strong, loss_val_strong = my_teacher(val_loader=dataloaders['val'],  teacher=model_ft, device=device, use_strong=True)
            acc1_val_base, loss_val_base = my_teacher(val_loader=dataloaders['val'],  teacher=model_ft, device=device, use_base=True)
            acc1_val_test, loss_val_test = my_teacher(val_loader=dataloaders['val'],  teacher=model_ft, device=device, use_base=False) # 主要是是使用False

            acc1_val_aug_weak, loss_val_aug_weak = my_teacher(val_loader=dataloaders['val_aug'],  teacher=model_ft, device=device, use_weak=True)
            acc1_val_aug_strong, loss_val_aug_strong = my_teacher(val_loader=dataloaders['val_aug'],  teacher=model_ft, device=device, use_strong=True)
            
            affinity_w_b = acc1_val_aug_weak / acc1_val_base
            affinity_s_b = acc1_val_aug_strong / acc1_val_base
            affinity_b_w = acc1_val_base / acc1_val_aug_weak
            affinity_b_s = acc1_val_base / acc1_val_aug_strong

            # diversity_w_b = loss_train_weak / loss_train_base
            # diversity_s_b = loss_train_strong / loss_train_base

        elif two_teacher or one_teacher:
            # acc1_train_weak, loss_train_weak = my_train(dataloaders['train'], model_ft, teacher, device, temp, alpha, use_weak=True)
            # acc1_train_strong, loss_train_strong = my_train(dataloaders['train'], model_ft, teacher, device, temp, alpha, use_strong=True)
            # acc1_train_base, loss_train_base = my_train(dataloaders['train'], model_ft, teacher, device, temp, alpha, use_base=True)
            
            acc1_val_base, loss_val_base, = my_validate(dataloaders['val'], model_ft, teacher, device,use_base=True)
            acc1_val_test, loss_val_test, = my_validate(dataloaders['val'], model_ft, teacher, device,use_base=False)

            acc1_val_aug_strong, loss_val_aug_strong = my_validate_aug(dataloaders['val_aug'], model_ft, teacher, device,use_strong=True)
            acc1_val_aug_weak, loss_val_aug_weak = my_validate_aug(dataloaders['val_aug'], model_ft, teacher, device,use_strong=False)

            affinity_w_b = acc1_val_aug_weak / acc1_val_base
            affinity_s_b = acc1_val_aug_strong / acc1_val_base
            affinity_b_w = acc1_val_base / acc1_val_aug_weak
            affinity_b_s = acc1_val_base / acc1_val_aug_strong

            # diversity_w_b = loss_train_weak / loss_train_base
            # diversity_s_b = loss_train_strong / loss_train_base

        else:
            assert False

        print_str = [
                    # f'Training complete in {hours:.0f}h {minutes:.0f}m {seconds:.0f}s\n',  
                    # f"Best validation accuracy is: {best_acc1}\n",
                    # f"at epoch: {best_epoch}\n",
                    # f"The accuracy gap between train and val is: {train_val_acc_gap}\n",
                    f'Model:{student_name}\n',

                    # f'acc1_train_base: {acc1_train_base}',
                    # f'acc1_train_weak: {acc1_train_weak}\n',
                    # f'acc1_train_strong: {acc1_train_strong}\n',
                    # f'acc1_train_base: {acc1_train_base}\n',

                    f'acc1_val_test: {acc1_val_test}\n',
                    f'acc1_val_base: {acc1_val_base}\n',
                    f'acc1_val_aug_strong: {acc1_val_aug_strong}\n',
                    f'acc1_val_aug_weak: {acc1_val_aug_weak}\n\n',

                    # f'loss_train_base: {loss_train_base}',

                    # f'loss_train_weak: {loss_train_weak}\n',
                    # f'loss_train_strong: {loss_train_strong}\n',
                    # f'loss_train_base: {loss_train_base}\n',

                    # f'loss_val_test: {loss_val_test}\n',
                    # f'loss_val_base: {loss_val_base}\n',
                    # f'loss_val_aug_strong: {loss_val_aug_strong}\n',
                    # f'loss_val_aug_weak: {loss_val_aug_weak}\n\n',


                    f'Afinity_w_b:{affinity_w_b}\n', 
                    f'Afinity_s_b:{affinity_s_b}\n', 
                    f'affinity_b_w:{affinity_b_w}\n', 
                    f'affinity_b_s:{affinity_b_s}\n', 

                    # f'diversity_w_b:{diversity_w_b}\n', 
                    # f'diversity_s_b:{diversity_s_b}\n', 
                    
                    ]
        print_write(print_str,file_name)
