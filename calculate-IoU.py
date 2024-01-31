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

from DataLoaderCIFAR import Load_CIFAR100
from DataLoaderImageNet import Load_ImageNet

# 主要计算T与S的IoU ———— 2024.01.01 好快啊2023就不再成为log的记录的符号了



data_root = {
        # 'CIFAR100': r'E:/cifar-100-python/clean_img',
        'CIFAR100': r'/mnt/d/data/cifar-100-python/clean_img',
        'CIFAR10': '/mnt/d/data/cifar-10-batches-py/clean_img',
        # 'ImageNet':'E:\ImageNet2012',
        'ImageNet':'/mnt/e/dataset/ImageNet/data/ImageNet2012',
        }

model_path = {
    # cifar100
    # 'T1w' : "run-teacher/CIFAR100_train_61_get-teacher-weak/checkpoint_bestAcc1.pth.tar",
    # 'T1s' : "run-teacher/CIFAR100_train_80_get-teacher-strong/checkpoint_bestAcc1.pth.tar",
    # 'T2w' : "run-teacher/CIFAR100_train_61_get-teacher-weak-2/checkpoint_bestAcc1.pth.tar",
    # 'T2s' : "run-teacher/CIFAR100_train_45_get-teacher-strong-2/checkpoint_bestAcc1.pth.tar",

    # "T1w_T2w_Sw":"run-cifar100/CIFAR100_distil_T1w_T2w_Sw/run-4-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2w_Ss":"run-cifar100/CIFAR100_distil_T1w_T2w_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_Sw":"run-cifar100/CIFAR100_distil_T1s_T2w_Sw/run-4-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_Ss":"run-cifar100/CIFAR100_distil_T1s_T2w_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_Sw":"run-cifar100/CIFAR100_distil_T1w_T2s_Sw/run-3-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_Ss":"run-cifar100/CIFAR100_distil_T1w_T2s_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_Sw":"run-cifar100/CIFAR100_distil_T1s_T2s_Sw/run-3-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_Ss":"run-cifar100/CIFAR100_distil_T1s_T2s_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar"

    # cifar100_imb100
    # 'T1w_T2w_Sw' : "epoch_175/CIFAR100_imb100_distil_T1w_T2w_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # 'T1w_T2w_Ss' : "epoch_175/CIFAR100_imb100_distil_T1w_T2w_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # 'T1s_T2w_Sw' : "epoch_175/CIFAR100_imb100_distil_T1s_T2w_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # 'T1s_T2w_Ss' : "epoch_175/CIFAR100_imb100_distil_T1s_T2w_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # 'T1w_T2s_Sw' : "epoch_175/CIFAR100_imb100_distil_T1w_T2s_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # 'T1w_T2s_Ss' : "epoch_175/CIFAR100_imb100_distil_T1w_T2s_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # 'T1s_T2s_Sw' : "epoch_175/CIFAR100_imb100_distil_T1s_T2s_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # 'T1s_T2s_Ss' : "epoch_175/CIFAR100_imb100_distil_T1s_T2s_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",

    # # ImageNet_LT
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

    # # # ImagNet 平衡完整的
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="计算两模型的激活图的IoU")
parser.add_argument("--m1_path",default="",#"run-teacher/CIFAR100_train_61_get-teacher-weak/checkpoint_bestAcc1.pth.tar",
                    help="模型1的加载路径。")
parser.add_argument("--m1_is_teacher",default=True,action='store_true',
                    help="模型1是否是教师模型。")
parser.add_argument("--m2_path",default="",#"11112-runs-save/CIFAR100_imb100_distil_300_T1s_T2s_Ss/checkpoint_bestAcc1.pth.tar",
                    help="模型2的加载路径。")
parser.add_argument("--m2_is_teacher",default=False,action='store_true',
                    help="模型2是否是教师模型。")
parser.add_argument("--dataset",default="CIFAR100_imb100",
                    help="所输入的数据集。")
# parser.add_argument("--phase",default="val",
#                     help="说要计算IoU的数据集。")
parser.add_argument('--S_add_weak',default=False, action='store_true',
                    help="S 学生是否添加弱增强。")
parser.add_argument('--S_add_strong',default=False, action='store_true',
                    help="S 学生是否添加强增强。")
parser.add_argument('--batch_size', default=128, type=int,
                    help='批次大小')
parser.add_argument('--add_name', default="01",
                    help='训练多后都不知道谁是谁了，再加个名字后缀来区分一下')


# newly added by zsw 上面没用了，别用，懒得删改了，将就用吧
modes = ['T1_T2','T1_S','T2_S']
parser.add_argument("--model_name",default="",
                    help="加载模型的名字。")
parser.add_argument("--mode",default="",choices=modes,
                    help="选择IoU计算的对象。")


def get_logfile_name(path):
    get_time = datetime.datetime.now().strftime('%b%d_%H-%M') # 月 日 时 分
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

    # 修改全连接层的输出维度
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, class_num)

    print("加载的模型路径是：", model_path)

    checkpoint = torch.load(model_path, map_location=device)
    resnet_model.load_state_dict(checkpoint['state_dict'])

    # 创建新的卷积层
    in_ch = resnet_model.fc.in_features
    out_ch = resnet_model.fc.out_features
    final_conv = nn.Conv2d(in_ch, out_ch, 1, 1)

    # 设置新卷积层的权重
    fc_weights = resnet_model.fc.state_dict()["weight"].view(out_ch, in_ch, 1, 1)
    fc_bias = resnet_model.fc.state_dict()["bias"]
    final_conv.load_state_dict({"weight": fc_weights, "bias": fc_bias})

    # 构建模型
    resnet_conv = nn.Sequential(*list(resnet_model.children())[:-2] + [final_conv])

    return resnet_model, resnet_conv


if __name__ == '__main__':
    args = parser.parse_args()

    if args.mode == 'T1_T2':
        args.m1_path = args.model_name.split('_')[0]
        args.m1_is_teacher = True
        args.m2_path = args.model_name.split('_')[1]
        args.m2_is_teacher = True
        if "Sw" in args.model_name:
            args.S_add_weak = True
            print("使用S_add_weak")
        elif "Ss" in args.model_name:
            args.S_add_strong = True
            print("使用S_add_strong")

    elif args.mode == 'T1_S': 
        args.m1_path = args.model_name.split('_')[0]
        args.m1_is_teacher = True
        args.m2_path = args.model_name
        args.m2_is_teacher = False
        if "Sw" in args.model_name:
            args.S_add_weak = True
            print("使用S_add_weak")
        elif "Ss" in args.model_name:
            args.S_add_strong = True
            print("使用S_add_strong")

    elif args.mode == 'T2_S': 
        args.m1_path = args.model_name.split('_')[1]
        args.m1_is_teacher = True
        args.m2_path = args.model_name
        args.m2_is_teacher = False
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
        print("加载load_CIFAR100函数")
        data = {x: Load_CIFAR100(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                        batch_size=args.batch_size, num_workers=4,
                        shuffle=True if x == 'train' else False)
        # for x in ['train', 'val']} 
        for x in ['train', 'val','val_aug']} 
        class_num = 100
    elif 'ImageNet' in args.dataset:
        print("加载ImageNet函数")
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

    path = f"./runs_IoU/{args.add_name}"      
    log_file = get_logfile_name(path=path)
    print_str = [
        f'--model1_path:{model_path[args.m1_path]}\n',f'--is_teahcer:{args.m1_is_teacher}\n'
        f'--model2_path:{model_path[args.m2_path]}\n',f'--is_teahcer:{args.m2_is_teacher}\n'
    ]
    print_write(print_str, log_file)

    for phase in ['train', 'val','val_aug']:
    # for phase in ['val_aug']:
        total_intersection = 0
        total_union = 0
        # phase = args.phase
        # total_batches = len(data[phase])
        # val: original None  trian：weak strong（aug01...aug08)
        for batch_idx, (imgs_weak,imgs_strong, labels) in enumerate(tqdm(data[phase])):
            imgs_weak,imgs_strong = imgs_weak.to(device),imgs_strong.to(device)
            with torch.no_grad():
                if phase == 'val':
                    model_output_cam1 = conv_1(imgs_weak)
                    model_output_cam2 = conv_2(imgs_weak)
                # elif phase == 'train': # 现在多了一个val_aug
                elif phase == 'train' or phase == 'val_aug':
                    if args.S_add_weak:
                        model_output_cam1 = conv_1(imgs_weak)
                        model_output_cam2 = conv_2(imgs_weak)
                    elif args.S_add_strong:
                        model_output_cam1 = conv_1(imgs_strong)
                        model_output_cam2 = conv_2(imgs_strong)
                        
            Threshold = 0.5
            model_cam1 = model_output_cam1.sum(dim=1) 
            model_cam2 = model_output_cam2.sum(dim=1)
            
            resized_cam1 = F.interpolate(model_cam1.unsqueeze(0), size=(imgs_weak.shape[2], imgs_weak.shape[3]), mode="bilinear", align_corners=False).squeeze(0)
            normalized_cam1 = torch.zeros_like(resized_cam1)
            for i in range(imgs_weak.size(0)):
                min_value = resized_cam1[i].min()
                max_value = resized_cam1[i].max()
                normalized_cam1[i] = (resized_cam1[i] - min_value) / (max_value - min_value + 1e-8)
            binary_cam1 = (normalized_cam1 > Threshold).long()


            resized_cam2 = F.interpolate(model_cam2.unsqueeze(0), size=(imgs_weak.shape[2], imgs_weak.shape[3]), mode="bilinear", align_corners=False).squeeze(0)
            normalized_cam2 = torch.zeros_like(resized_cam2)
            for i in range(imgs_weak.size(0)):
                min_value = resized_cam2[i].min()
                max_value = resized_cam2[i].max()
                normalized_cam2[i] = (resized_cam2[i] - min_value) / (max_value - min_value + 1e-8)
            binary_cam2 = (normalized_cam2 > Threshold).long()
            
            # Compute IoU for the current batch
            intersection = torch.sum(binary_cam1 * binary_cam2) #交集 白色部分
            union = torch.sum((binary_cam1 + binary_cam2) > 0) # 并集
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
        if phase == 'val': # 还有val_aug怎么整呢
            add_name = "None"
        print_str = [
            f"{args.dataset}_{phase}_{add_name}: {average_iou}\n"
            ]
        print_write(print_str,log_file)



# 运行语句例子
# python 计算IoU.py --m1_path run-teacher/CIFAR100_train_61_get-teacher-weak/checkpoint_bestAcc1.pth.tar \
# --m1_is_teacher \
# --m2_path 11112-runs-save/CIFAR100_imb100_distil_300_T1s_T2s_Ss/checkpoint_bestAcc1.pth.tar \
# --S_add_weak
# --add_name T1—T1s_T2s_Ss


# # T1w_T2w_Sw 基线
# python 计算IoU.py \
#     --m1_path T1w --m1_is_teacher \
#     --m2_path T2w --m2_is_teacher \
#     --S_add_weak \
#     --dataset ImageNet_LT \
#     --add_name T1w_T2w



# 3.0 版本
# # T1w_T2w_Sw 基线
# python 计算IoU.py \
#     --mode T1_T2 \
#     --model_name T1w_T2w_Sw \
#     --dataset ImageNet_LT \
#     --add_name T1w_T2w