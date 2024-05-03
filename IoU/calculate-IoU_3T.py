import argparse
import datetime
import torch
from torch import nn as nn
from torchvision import *
from PIL import Image as PILImage
import cv2
import json
import numpy as np
import torchvision.transforms as transforms
from fastai import *
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import torch.nn.functional as F
import sys
sys.path.append("data_loader")
from DataLoaderCIFAR import Load_CIFAR100
from DataLoaderImageNet import Load_ImageNet


data_root = {
        # 'CIFAR100': r'E:/cifar-100-python/clean_img',
        'CIFAR100': '/mnt/d/data/cifar-100-python/clean_img',
        'CIFAR10': '/mnt/d/data/cifar-10-batches-py/clean_img',
        # 'ImageNet':'E:\ImageNet2012',
        'ImageNet':'/mnt/e/dataset/ImageNet/data/ImageNet2012',
        }

model_path = {
    # cifar100_imb
    'T1w' : "run-teacher/CIFAR100_train_61_get-teacher-weak/checkpoint_bestAcc1.pth.tar",
    'T1s' : "run-teacher/CIFAR100_train_80_get-teacher-strong/checkpoint_bestAcc1.pth.tar",
    'T2w' : "run-teacher/CIFAR100_train_61_get-teacher-weak-2/checkpoint_bestAcc1.pth.tar",
    'T2s' : "run-teacher/CIFAR100_train_45_get-teacher-strong-2/checkpoint_bestAcc1.pth.tar",
    'T3w' : "run-teacher/CIFAR100_train_get-Tw-3/run-1-epoch45/checkpoint_bestAcc1.pth.tar",
    'T3s' : "run-teacher/CIFAR100_train_get-Ts-3/run-1-epoch45/checkpoint_bestAcc1.pth.tar",
    
    # cifar100_imb
    # "T1w_T2w_T3w_Sw" : "runs-3T/CIFAR100_imb100_distil_T1w_T2w_T3w_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2w_T3w_Ss" : "runs-3T/CIFAR100_imb100_distil_T1w_T2w_T3w_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_T3w_Sw" : "runs-3T/CIFAR100_imb100_distil_T1s_T2w_T3w_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_T3w_Sw" : "runs-3T/CIFAR100_imb100_distil_T1w_T2s_T3w_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_T3w_Ss" : "runs-3T/CIFAR100_imb100_distil_T1s_T2w_T3w_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_T3w_Ss" : "runs-3T/CIFAR100_imb100_distil_T1w_T2s_T3w_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2w_T3s_Ss" : "runs-3T/CIFAR100_imb100_distil_T1w_T2w_T3s_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_T3w_Sw" : "runs-3T/CIFAR100_imb100_distil_T1s_T2s_T3w_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_T3s_Sw" : "runs-3T/CIFAR100_imb100_distil_T1w_T2s_T3s_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_T3s_Sw" : "runs-3T/CIFAR100_imb100_distil_T1s_T2w_T3s_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_T3w_Ss" : "runs-3T/CIFAR100_imb100_distil_T1s_T2s_T3w_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_T3s_Ss" : "runs-3T/CIFAR100_imb100_distil_T1w_T2s_T3s_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_T3s_Ss" : "runs-3T/CIFAR100_imb100_distil_T1s_T2w_T3s_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_T3s_Sw" : "runs-3T/CIFAR100_imb100_distil_T1s_T2s_T3s_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_T3s_Ss" : "runs-3T/CIFAR100_imb100_distil_T1s_T2s_T3s_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",

    # cifar100 balance
    "T1w_T2w_T3w_Sw" : "runs-3T-cifar100-balance/CIFAR100_distil_T1w_T2w_T3w_Sw/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1w_T2w_T3w_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1w_T2w_T3w_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1s_T2w_T3w_Sw" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2w_T3w_Sw/run-1-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1w_T2s_T3w_Sw" : "runs-3T-cifar100-balance/CIFAR100_distil_T1w_T2s_T3w_Sw/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1s_T2w_T3w_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2w_T3w_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1w_T2s_T3w_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1w_T2s_T3w_Ss/run-3-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1w_T2w_T3s_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1w_T2w_T3s_Ss/run-1-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1s_T2s_T3w_Sw" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2s_T3w_Sw/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1w_T2s_T3s_Sw" : "runs-3T-cifar100-balance/CIFAR100_distil_T1w_T2s_T3s_Sw/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1s_T2w_T3s_Sw" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2w_T3s_Sw/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1s_T2s_T3w_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2s_T3w_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1w_T2s_T3s_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1w_T2s_T3s_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1s_T2w_T3s_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2w_T3s_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1s_T2s_T3s_Sw" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2s_T3s_Sw/run-3-epoch200/checkpoint_bestAcc1.pth.tar",
    "T1s_T2s_T3s_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2s_T3s_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",

}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Calculate the Intersection over Union (IoU) of activation maps between the two models.")
parser.add_argument("--m1_path",default="",#"run-teacher/CIFAR100_train_61_get-teacher-weak/checkpoint_bestAcc1.pth.tar",
                    help="")
parser.add_argument("--m1_is_teacher",default=True,action='store_true',
                    help="")
parser.add_argument("--m2_path",default="",#"11112-runs-save/CIFAR100_imb100_distil_300_T1s_T2s_Ss/checkpoint_bestAcc1.pth.tar",
                    help="")
parser.add_argument("--m2_is_teacher",default=False,action='store_true',
                    help="")
parser.add_argument("--m3_path",default="",#"11112-runs-save/CIFAR100_imb100_distil_300_T1s_T2s_Ss/checkpoint_bestAcc1.pth.tar",
                    help="")
parser.add_argument("--m3_is_teacher",default=False,action='store_true',
                    help="")
parser.add_argument("--dataset",default="CIFAR100_imb100",
                    help="")

parser.add_argument('--S_add_weak',default=False, action='store_true',
                    help="")
parser.add_argument('--S_add_strong',default=False, action='store_true',
                    help="")
parser.add_argument('--batch_size', default=128, type=int,
                    help='')
parser.add_argument('--add_name', default="01",
                    help='')


# 
modes = ['T_T','T12_S','T23_S','T13_S']
parser.add_argument("--model_name",default="",
                    help="The name of the loaded model.")
parser.add_argument("--mode",default="",choices=modes,
                    help="Select the objects for IoU calculation.")


def get_logfile_name(path):
    get_time = datetime.datetime.now().strftime('%b%d_%H-%M') 
    # file_name = get_time + '_log.txt'
    file_name ='calculate_IoU_log.txt'
    if not os.path.exists(path):  
        os.makedirs(path)  
        
    return os.path.join(path, file_name)


def print_write(print_str, log_file):
    print(*print_str)
    with open(log_file, 'a') as f:
        print(*print_str, file=f)

def load_model(model_path=None, class_num=None, is_teacher=False, device=None):
    if is_teacher:
        resnet_model = models.resnet50(pretrained=False) 
    else:
        resnet_model = models.resnet18(pretrained=False)

    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, class_num)

    print("The loaded model's path is:", model_path)

    checkpoint = torch.load(model_path, map_location=device)
    resnet_model.load_state_dict(checkpoint['state_dict'])


    in_ch = resnet_model.fc.in_features
    out_ch = resnet_model.fc.out_features
    final_conv = nn.Conv2d(in_ch, out_ch, 1, 1)

    fc_weights = resnet_model.fc.state_dict()["weight"].view(out_ch, in_ch, 1, 1)
    fc_bias = resnet_model.fc.state_dict()["bias"]
    final_conv.load_state_dict({"weight": fc_weights, "bias": fc_bias})


    resnet_conv = nn.Sequential(*list(resnet_model.children())[:-2] + [final_conv])

    return resnet_model, resnet_conv


if __name__ == '__main__':
    args = parser.parse_args()

    if args.mode == 'T_T':
        args.m1_path = args.model_name.split('_')[0]
        args.m1_is_teacher = True
        args.m2_path = args.model_name.split('_')[1]
        args.m2_is_teacher = True
        args.m3_path = args.model_name.split('_')[2]
        args.m3_is_teacher = True
        if "Sw" in args.model_name:
            args.S_add_weak = True

        elif "Ss" in args.model_name:
            args.S_add_strong = True


    elif args.mode == 'T12_S': 
        args.m1_path = args.model_name.split('_')[0]
        args.m1_is_teacher = True
        args.m2_path = args.model_name.split('_')[1]
        args.m2_is_teacher = True
        args.m3_path = args.model_name
        args.m3_is_teacher = False
        if "Sw" in args.model_name:
            args.S_add_weak = True

        elif "Ss" in args.model_name:
            args.S_add_strong = True


    elif args.mode == 'T23_S': 
        args.m1_path = args.model_name.split('_')[1]
        args.m1_is_teacher = True
        args.m2_path = args.model_name.split('_')[2]
        args.m2_is_teacher = True
        args.m3_path = args.model_name
        args.m3_is_teacher = False
        if "Sw" in args.model_name:
            args.S_add_weak = True
            print("使用S_add_weak")
        elif "Ss" in args.model_name:
            args.S_add_strong = True
            print("使用S_add_strong")
    
    else:
        print(args.mode)
        assert False



    if 'CIFAR100' in args.dataset:
        data = {x: Load_CIFAR100(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                        batch_size=args.batch_size, num_workers=4,
                        shuffle=True if x == 'train' else False)
        # for x in ['train', 'val']} 
        for x in ['train', 'val','val_aug']} 
        class_num = 100
    elif 'ImageNet' in args.dataset:
        data = {x: Load_ImageNet(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                        batch_size=args.batch_size, num_workers=4,
                        shuffle=True if x == 'train' else False)
        # for x in ['train', 'val']} 
        for x in ['train', 'val','val_aug']} 
        class_num = 1000
    else:
        assert False

    model_1, conv_1 = load_model(model_path=model_path[args.m1_path], class_num=class_num, is_teacher=args.m1_is_teacher)
    model_1 = model_1.to(device)
    conv_1 = conv_1.to(device)
    model_1.eval()
    conv_1.eval()

    model_2, conv_2 = load_model(model_path=model_path[args.m2_path], class_num=class_num, is_teacher=args.m2_is_teacher)
    model_2 = model_2.to(device)
    conv_2 = conv_2.to(device)
    model_2.eval()
    conv_2.eval()

    model_3, conv_3 = load_model(model_path=model_path[args.m3_path], class_num=class_num, is_teacher=args.m3_is_teacher)
    model_3 = model_3.to(device)
    conv_3 = conv_3.to(device)
    model_3.eval()
    conv_3.eval()

    path = f"./runs_IoU/{args.add_name}"      
    log_file = get_logfile_name(path=path)
    print_str = [
        f'--model1_path:{model_path[args.m1_path]}\n',f'--is_teahcer:{args.m1_is_teacher}\n'
        f'--model2_path:{model_path[args.m2_path]}\n',f'--is_teahcer:{args.m2_is_teacher}\n'
        f'--model3_path:{model_path[args.m3_path]}\n',f'--is_teahcer:{args.m3_is_teacher}\n'
    ]
    print_write(print_str, log_file)

    for phase in ['train', 'val', 'val_aug']:
        total_intersection = 0
        total_union = 0

        for batch_idx, (imgs_weak, imgs_strong, labels) in enumerate(tqdm(data[phase])):
            imgs_weak, imgs_strong = imgs_weak.to(device), imgs_strong.to(device)

            with torch.no_grad():
                if phase == 'val':
                    model_output_cam1 = conv_1(imgs_weak)
                    model_output_cam2 = conv_2(imgs_weak)
                    model_output_cam3 = conv_3(imgs_weak)
                elif phase == 'train' or phase == 'val_aug':
                    if args.S_add_weak:
                        model_output_cam1 = conv_1(imgs_weak)
                        model_output_cam2 = conv_2(imgs_weak)
                        model_output_cam3 = conv_3(imgs_weak)
                    elif args.S_add_strong:
                        model_output_cam1 = conv_1(imgs_strong)
                        model_output_cam2 = conv_2(imgs_strong)
                        model_output_cam3 = conv_3(imgs_strong)

            Threshold = 0.4

            model_cam1 = model_output_cam1.sum(dim=1)
            model_cam2 = model_output_cam2.sum(dim=1)
            model_cam3 = model_output_cam3.sum(dim=1)

            # Process model_cam1
            resized_cam1 = F.interpolate(model_cam1.unsqueeze(0), size=(imgs_weak.shape[2], imgs_weak.shape[3]),
                                        mode="bilinear", align_corners=False).squeeze(0)
            normalized_cam1 = torch.zeros_like(resized_cam1)
            for i in range(imgs_weak.size(0)):
                min_value = resized_cam1[i].min()
                max_value = resized_cam1[i].max()
                normalized_cam1[i] = (resized_cam1[i] - min_value) / (max_value - min_value + 1e-8)
            binary_cam1 = (normalized_cam1 > Threshold).long()

            # Process model_cam2
            resized_cam2 = F.interpolate(model_cam2.unsqueeze(0), size=(imgs_weak.shape[2], imgs_weak.shape[3]),
                                        mode="bilinear", align_corners=False).squeeze(0)
            normalized_cam2 = torch.zeros_like(resized_cam2)
            for i in range(imgs_weak.size(0)):
                min_value = resized_cam2[i].min()
                max_value = resized_cam2[i].max()
                normalized_cam2[i] = (resized_cam2[i] - min_value) / (max_value - min_value + 1e-8)
            binary_cam2 = (normalized_cam2 > Threshold).long()

            # Process model_cam3
            resized_cam3 = F.interpolate(model_cam3.unsqueeze(0), size=(imgs_weak.shape[2], imgs_weak.shape[3]),
                                        mode="bilinear", align_corners=False).squeeze(0)
            normalized_cam3 = torch.zeros_like(resized_cam3)
            for i in range(imgs_weak.size(0)):
                min_value = resized_cam3[i].min()
                max_value = resized_cam3[i].max()
                normalized_cam3[i] = (resized_cam3[i] - min_value) / (max_value - min_value + 1e-8)
            binary_cam3 = (normalized_cam3 > Threshold).long()

            # Compute IoU for the current batch
            intersection = torch.sum(binary_cam1 * binary_cam2 * binary_cam3) 
            union = torch.sum((binary_cam1 + binary_cam2 + binary_cam3) > 0) 
            iou = intersection / union

            # Accumulate intersection and union for the entire dataset
            total_intersection += iou.item() * len(imgs_weak)
            total_union += len(imgs_weak)

        average_iou = total_intersection / total_union

        # f"data augmentation --> weak: {args.S_add_weak} —— strong: {args.S_add_strong}\n"
        if args.S_add_weak:
            add_name = "weak"
        if args.S_add_strong:
            add_name = "strong"
        if phase == 'val': 
            add_name = "None"
        print_str = [
            f"{args.dataset}_{phase}_{add_name}: {average_iou}\n"
            ]
        print_write(print_str,log_file)


