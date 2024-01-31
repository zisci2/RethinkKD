import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_squared_error
import numpy as np
from torchvision import *
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from sklearn.calibration import calibration_curve

# from DataLoaderCIFAR_clean import Load_CIFAR100
from DataLoaderCIFAR import Load_CIFAR100
from DataLoaderImageNet import Load_ImageNet
#TODO 本周(2023.12.19)要做
# 1）计算ECE --> ece论文的图需要画吗，先将画图需要的数据记录下来吧
# 2）
'''
对了，在计算ECE的时候，val和val_aug可以都是一下，也别忘了保存一下prediction probability
（也就是我们要的confidence）和对应的accuracy @风之谷 
'''

data_root = {
        # 'CIFAR100': r'E:/cifar-100-python/clean_img',
        'CIFAR100': r'/mnt/d/data/cifar-100-python/clean_img',
        'CIFAR10': '/mnt/d/data/cifar-10-batches-py/clean_img',
        # 'ImageNet':'E:\ImageNet2012',
        'ImageNet':'/mnt/e/dataset/ImageNet/data/ImageNet2012',
        }

# teacher_path = {
models_path = {
    # # cifar100_LT
    "T1w" : "run-teacher/CIFAR100_train_61_get-teacher-weak/checkpoint_bestAcc1.pth.tar",
    "T2w" : "run-teacher/CIFAR100_train_61_get-teacher-weak-2/checkpoint_bestAcc1.pth.tar",
    "T1s" : "run-teacher/CIFAR100_train_80_get-teacher-strong/checkpoint_bestAcc1.pth.tar",
    "T2s" : "run-teacher/CIFAR100_train_45_get-teacher-strong-2/checkpoint_bestAcc1.pth.tar",
    # #cifar100_LT
    "only_Sw" : "11111-runs-save/CIFAR100_imb100_train_90_only_Sw/checkpoint_bestAcc1.pth.tar",
    "only_Ss" : "11111-runs-save/CIFAR100_imb100_train_75_only_Ss/checkpoint_bestAcc1.pth.tar",
    "T1w_Sw" : "epoch_175_old/CIFAR100_imb100_distil_175_T1w_Sw/checkpoint_bestAcc1.pth.tar",
    "T1w_Ss" : "epoch_175_old/CIFAR100_imb100_distil_175_T1w_Ss/checkpoint_bestAcc1.pth.tar",
    "T1s_Sw" : "epoch_175_old/CIFAR100_imb100_distil_175_T1s_Sw/checkpoint_bestAcc1.pth.tar",
    "T1s_Ss" : "epoch_175_old/CIFAR100_imb100_distil_175_T1s_Ss/checkpoint_bestAcc1.pth.tar",

    "T1w_T2w_Sw" : "epoch_175/CIFAR100_imb100_distil_T1w_T2w_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    "T1w_T2w_Ss" : "epoch_175/CIFAR100_imb100_distil_T1w_T2w_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",

    "T1s_T2w_Sw" : "epoch_175/CIFAR100_imb100_distil_T1s_T2w_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    "T1s_T2w_Ss" : "epoch_175/CIFAR100_imb100_distil_T1s_T2w_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    "T1w_T2s_Sw" : "epoch_175/CIFAR100_imb100_distil_T1w_T2s_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    "T1w_T2s_Ss" : "epoch_175/CIFAR100_imb100_distil_T1w_T2s_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",

    # "T1s_T2s_Sw" : "epoch_175/CIFAR100_imb100_distil_T1s_T2s_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_Ss" : "epoch_175/CIFAR100_imb100_distil_T1s_T2s_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",

    # ciar100
    # "T1w_Sw" : "run-cifar100/CIFAR100_distil_T1w_Sw/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_Ss" : "run-cifar100/CIFAR100_distil_T1s_Ss/run-3-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1w_Ss" : "run-cifar100/CIFAR100_distil_T1w_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_Sw" : "run-cifar100/CIFAR100_distil_T1s_Sw/run-2-epoch200/checkpoint_bestAcc1.pth.tar",

    # "T1w_T2w_Sw" : "run-cifar100/CIFAR100_distil_T1w_T2w_Sw/run-4-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_Sw" : "run-cifar100/CIFAR100_distil_T1s_T2w_Sw/run-4-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_Sw" : "run-cifar100/CIFAR100_distil_T1w_T2s_Sw/run-3-epoch200/checkpoint_bestAcc1.pth.tar",

    # "T1s_T2s_Ss" : "run-cifar100/CIFAR100_distil_T1s_T2s_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_Ss" : "run-cifar100/CIFAR100_distil_T1w_T2s_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_Ss" : "run-cifar100/CIFAR100_distil_T1s_T2w_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",

    # "T1w_T2w_Ss" : "run-cifar100/CIFAR100_distil_T1w_T2w_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_Sw" : "run-cifar100/CIFAR100_distil_T1s_T2s_Sw/run-3-epoch200/checkpoint_bestAcc1.pth.tar",

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


    # cifar100 3T
    # 'T1w' : "run-teacher/CIFAR100_train_61_get-teacher-weak/checkpoint_bestAcc1.pth.tar",
    # 'T1s' : "run-teacher/CIFAR100_train_80_get-teacher-strong/checkpoint_bestAcc1.pth.tar",
    # 'T2w' : "run-teacher/CIFAR100_train_61_get-teacher-weak-2/checkpoint_bestAcc1.pth.tar",
    # 'T2s' : "run-teacher/CIFAR100_train_45_get-teacher-strong-2/checkpoint_bestAcc1.pth.tar",
    # 'T3w' : "run-teacher/CIFAR100_train_get-Tw-3/run-1-epoch45/checkpoint_bestAcc1.pth.tar",
    # 'T3s' : "run-teacher/CIFAR100_train_get-Ts-3/run-1-epoch45/checkpoint_bestAcc1.pth.tar",
    
    # cifar100_imb 3T
    # "T1w_T2w_T3w_Sw" : "runs-3T-cifar100_imb100/CIFAR100_imb100_distil_T1w_T2w_T3w_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2w_T3w_Ss" : "runs-3T-cifar100_imb100/CIFAR100_imb100_distil_T1w_T2w_T3w_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_T3w_Sw" : "runs-3T-cifar100_imb100/CIFAR100_imb100_distil_T1s_T2w_T3w_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_T3w_Sw" : "runs-3T-cifar100_imb100/CIFAR100_imb100_distil_T1w_T2s_T3w_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_T3w_Ss" : "runs-3T-cifar100_imb100/CIFAR100_imb100_distil_T1s_T2w_T3w_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_T3w_Ss" : "runs-3T-cifar100_imb100/CIFAR100_imb100_distil_T1w_T2s_T3w_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2w_T3s_Ss" : "runs-3T-cifar100_imb100/CIFAR100_imb100_distil_T1w_T2w_T3s_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_T3w_Sw" : "runs-3T-cifar100_imb100/CIFAR100_imb100_distil_T1s_T2s_T3w_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar", # # 有问题
    # "T1w_T2s_T3s_Sw" : "runs-3T-cifar100_imb100/CIFAR100_imb100_distil_T1w_T2s_T3s_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_T3s_Sw" : "runs-3T-cifar100_imb100/CIFAR100_imb100_distil_T1s_T2w_T3s_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_T3w_Ss" : "runs-3T-cifar100_imb100/CIFAR100_imb100_distil_T1s_T2s_T3w_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_T3s_Ss" : "runs-3T-cifar100_imb100/CIFAR100_imb100_distil_T1w_T2s_T3s_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_T3s_Ss" : "runs-3T-cifar100_imb100/CIFAR100_imb100_distil_T1s_T2w_T3s_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_T3s_Sw" : "runs-3T-cifar100_imb100/CIFAR100_imb100_distil_T1s_T2s_T3s_Sw/run-1-epoch175/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_T3s_Ss" : "runs-3T-cifar100_imb100/CIFAR100_imb100_distil_T1s_T2s_T3s_Ss/run-1-epoch175/checkpoint_bestAcc1.pth.tar",


    # # cifar100 balance
    # "T1w_T2w_T3w_Sw" : "runs-3T-cifar100-balance/CIFAR100_distil_T1w_T2w_T3w_Sw/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2w_T3w_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1w_T2w_T3w_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_T3w_Sw" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2w_T3w_Sw/run-1-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_T3w_Sw" : "runs-3T-cifar100-balance/CIFAR100_distil_T1w_T2s_T3w_Sw/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_T3w_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2w_T3w_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_T3w_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1w_T2s_T3w_Ss/run-3-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2w_T3s_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1w_T2w_T3s_Ss/run-1-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_T3w_Sw" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2s_T3w_Sw/run-2-epoch200/checkpoint_bestAcc1.pth.tar", # #有问题
    # "T1w_T2s_T3s_Sw" : "runs-3T-cifar100-balance/CIFAR100_distil_T1w_T2s_T3s_Sw/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_T3s_Sw" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2w_T3s_Sw/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_T3w_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2s_T3w_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1w_T2s_T3s_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1w_T2s_T3s_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2w_T3s_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2w_T3s_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_T3s_Sw" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2s_T3s_Sw/run-3-epoch200/checkpoint_bestAcc1.pth.tar",
    # "T1s_T2s_T3s_Ss" : "runs-3T-cifar100-balance/CIFAR100_distil_T1s_T2s_T3s_Ss/run-2-epoch200/checkpoint_bestAcc1.pth.tar",


    # # ImagNet 平衡完整的
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

parser = argparse.ArgumentParser(description='Distillation from resnet50')
parser.add_argument('--M',default='T1w_T2w_Sw',
                    help="加载的模型，教师或者学生模型")
parser.add_argument('--is_teacher',default=False,action='store_true')
parser.add_argument('--D',default='CIFAR100_imb100',
                    help='加载是数据集')
parser.add_argument('--batch_size',default=128)
parser.add_argument('--phase',default=None,
                    help='所使用的是哪个阶段的数据') # train val val_aug
parser.add_argument('--n_bins',default=10,
                    help='划分的格子数目')


def get_logfile_name(path,name,judge=False):
    get_time = datetime.datetime.now().strftime('%b%d_%H-%M') # 月 日 时 分
    # file_name = get_time + '_log.txt'
    if judge:
        file_name = f'Model_total.txt'
    else:
        file_name = f'{name}_ECE.txt'
    if not os.path.exists(path):  
        os.makedirs(path)  
        
    return os.path.join(path, file_name)

def print_write(print_str, log_file):
    print(*print_str)
    with open(log_file, 'a') as f:
        print(*print_str, file=f)

def ece_score(predict_output, label, log_file, n_bins=10):
    predict_output = torch.nn.functional.softmax(predict_output, dim=1)
    predict_index = torch.argmax(predict_output, dim=1)
    predict_value = predict_output[torch.arange(len(predict_output)), predict_index]
    predict_value_1, _ = torch.max(predict_output, dim=1)

    if torch.equal(predict_value,predict_value_1):
        print("很棒，是一样的")
    else:
        print("不行，不是一样的")

    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    acc_list = []
    conf_list = []
    x_coordinate = []
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        indices = (predict_value > a) & (predict_value <= b)
        Bm[m] = indices.sum().item()
        if Bm[m] != 0: # 确保区间有样本
            correct_indices = indices & (predict_index == label)
            acc[m] = correct_indices.sum().item() / Bm[m] # 预测对的
            conf[m] = predict_value[indices].sum().item() / Bm[m] # # 平均预测概率
            acc_list.append(acc[m])
            conf_list.append(conf[m])
            x_coordinate.append(b)
            print_str = ["{:.2f}-{:.2f} Bm[{}]:{:5d} correct:{:5d}     predict_all:{:.4f}     acc:{:.4f}     conf:{:.4f}".format(a, b,m, int(Bm[m]), int(correct_indices.sum().item()), predict_value[indices].sum().item(), acc[m], conf[m])]
        else:
            acc_list.append(int(0))
            conf_list.append(int(0))
            x_coordinate.append(b)
            # print("ece_score Bm -->",Bm)
            # print("ece_score acc[m] -->",acc[m])
            # print("ece_score conf[m] -->",conf[m])
            print_str = ["{}-{} Bm[{}]:{:5d}    correct:0         predict_all:0     acc:0     conf:0".format(a, b, m, int(Bm[m]))]
        print_write(print_str, log_file)
    # print_str = [f"n_bins --> {n_bins} Bm --> {list(Bm)}"]
    # print_write(print_str,log_file)
    ece = np.abs(acc - conf).mean()
    return ece,acc_list,conf_list,x_coordinate


def ece_score_numpy(predict_output, label, n_bins=10):
    # 转成概率
    predict_output = torch.nn.functional.softmax(predict_output, dim=1)
    predict_value_1, predict_index_1 = torch.max(predict_output, dim=1)
    
    predict_output = np.array(predict_output.cpu().numpy())
    label = np.array(label.cpu().numpy())
    if label.ndim > 1:
        label = np.argmax(label, axis=1)
    predict_index = np.argmax(predict_output, axis=1)
    predict_value = []
    for i in range(predict_output.shape[0]):
        predict_value.append(predict_output[i, predict_index[i]])
    predict_value = np.array(predict_value)

    # # 检查 predict_index 与 predict_index_1 是否相等
    # if np.array_equal(predict_index_1.cpu().numpy(), predict_index):
    #     print("predict_index and predict_index_1 are the same.")
    # else:
    #     print("predict_index and predict_index_1 are different.")

    # # 检查 predict_value 与 predict_value_1 是否数值相同
    # if np.allclose(predict_value_1.cpu().numpy(), predict_value):
    #     print("predict_value and predict_value_1 are numerically the same.")
    # else:
    #     print("predict_value and predict_value_1 are numerically different.")

    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(predict_output.shape[0]):
            if predict_value[i] > a and predict_value[i] <= b:
                Bm[m] += 1
                if predict_index[i] == label[i]: # 记录正确样本
                    acc[m] += 1
                conf[m] += predict_value[i]
        if Bm[m] != 0: #这个区间是否有样本
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
            print("ece_score_numpy Bm -->",Bm)
            print("ece_score_numpy acc[m] -->",acc[m])
            print("ece_score_numpy conf[m] -->",conf[m])
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)

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

    return resnet_model



def main(log_file,args):
    log_file_2 = get_logfile_name(path=f"./confidenc_ECE/n_bins-{args.n_bins}",name = args.M, judge=True)

    if "CIFAR" in args.D:
        class_num = 100
    elif "ImageNet" in args.D:
        class_num = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path=models_path[args.M],class_num=class_num,is_teacher=args.is_teacher,device=device)
    model = model.to(device=device)
    # 加载数据
    dataset = args.D
    batch_size = args.batch_size
    # data = {x: Load_CIFAR100(data_root=data_root[dataset.split("_")[0]], dataset=dataset, phase=x,
    #                 batch_size=batch_size, num_workers=4,
    #                 shuffle=True if x == 'train' else False)
    # for x in ['train','val','val_aug']} 
    if 'CIFAR100' in dataset:
        data = {x: Load_CIFAR100(data_root=data_root[dataset.split("_")[0]], dataset=dataset, phase=x,
                        batch_size=batch_size, num_workers=4,
                        shuffle=False)
        for x in ['train', 'val']} 
        class_num = 100
    elif "ImageNet" in dataset :
        data = {x: Load_ImageNet(data_root=data_root[dataset.split("_")[0]], dataset=dataset, phase=x,
                        batch_size=batch_size, num_workers=4,
                        shuffle=False)
        for x in ['train', 'val']} 
        class_num = 1000

    # 模型推理并获取预测概率
    model.eval()
    result = {
        "train":   {"ece": 0.0, "x_coordinate": None, "conf_list": None, "acc_list": None},
        "val":     {"ece": 0.0, "x_coordinate": None, "conf_list": None, "acc_list": None},
        "val_aug": {"ece": 0.0, "x_coordinate": None, "conf_list": None, "acc_list": None}
            }

    # for x in ['train','val','val_aug']:
    for x in ['train']:
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, (imgs_weak, imgs_strong, labels) in enumerate(tqdm(data[x])):
            # for batch_idx, (img_base, img_weak, img_strong, name, label) in enumerate(tqdm(data[phase])): # clean 特供版
                # # 要不全部的增强都试验一下吧，反正也耗不了多少时间
                if x == 'val_aug' or x == 'train':
                    if "Ss" in args.M:
                        inputs, labels = imgs_strong.to(device), labels.to(device)
                    else:
                        inputs, labels = imgs_weak.to(device), labels.to(device)
                else:
                    inputs, labels = imgs_weak.to(device), labels.to(device)
                
                #     # weak
                #     # strong
                
                outputs = model(inputs)
                all_outputs.append(outputs)
                all_labels.append(labels)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # ece,acc_list,conf_list,x_coordinate = ece_score(all_outputs, all_labels, n_bins=10)
        result[x]["ece"],result[x]["acc_list"],result[x]["conf_list"],result[x]["x_coordinate"] = ece_score(all_outputs, all_labels,log_file, n_bins=args.n_bins)
        # for i in range(len(result[x]["x_coordinate"])):
        #     print_str = [f'B:{result[x]["x_coordinate"][i]} confidence:{result[x]["x_coordinate"][i]} acc:{result[x]["acc_list"][i]}']
        #     print_write(print_str, log_file)
        print_str = [f"{args.M}-{args.D}-{x} (ECE): {result[x]['ece']}\n"]
        print_write(print_str, log_file)
        print_write(print_str, log_file_2)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    fig, axs = plt.subplots(1,3,figsize=(10, 4))
    # for i,x in enumerate(['train','val','val_aug']):
    for i,x in enumerate(['train']):
        axs[i].plot(result[x]["x_coordinate"],result[x]["conf_list"])
        axs[i].plot(result[x]["x_coordinate"],result[x]["acc_list"])
        axs[i].legend([f'conf',f'acc'])
        axs[i].set_xlabel('')
        axs[i].set_ylabel('Accuracy')
        axs[i].set_title(f"{x}")
        # 设置 y 轴标签格式化器
        axs[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        axs[i].ticklabel_format(style='plain', axis='y')  # 确保刻度标签没有偏移


    # ax.legend(['train','val','val_aug'])
    plt.suptitle(args.M)
    # plt.xlabel('')
    # plt.ylabel('Accuracy %')
    plt.tight_layout()
    # fig.savefig(f"./confidenc_ECE/{args.M}_{result[x]['ece']}.jpg")
    fig.savefig(f"./confidenc_ECE/{args.M}.jpg")
    # plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    log_file = get_logfile_name(path=f"./confidenc_ECE/n_bins-{args.n_bins}",name = args.M)
    main(log_file,args)






# python 计算ECE.py --M T-w


# import matplotlib.pyplot as plt
# from sklearn.calibration import calibration_curve
# import numpy as np

# def confidence_histograms(predict_output, label, n_bins=10):
#     predict_output = np.array(predict_output)
#     label = np.array(label)
#     if label.ndim > 1:
#         label = np.argmax(label, axis=1)
#     predict_index = np.argmax(predict_output, axis=1)
#     predict_value = predict_output[np.arange(len(predict_index)), predict_index]

#     plt.hist(predict_value, range=(0, 1), bins=n_bins, histtype='step', color='blue', lw=2)
#     plt.title('Confidence Histograms')
#     plt.xlabel('Predicted Probability')
#     plt.ylabel('Frequency')
#     plt.show()

# def reliability_diagrams(predict_output, label, n_bins=10):
#     prob_true, prob_pred = calibration_curve(label, predict_output, n_bins=n_bins, strategy='uniform')

#     plt.plot(prob_pred, prob_true, marker='o', color='blue', label='Model')
#     plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
#     plt.title('Reliability Diagram')
#     plt.xlabel('Mean Predicted Probability')
#     plt.ylabel('Fraction of Positives')
#     plt.legend()
#     plt.show()

# # 示例数据
# predict_output = np.random.rand(1000, 3)
# label = np.eye(3)[np.random.choice(3, 1000)]

# # 绘制 Confidence Histograms
# confidence_histograms(predict_output, label, n_bins=10)

# # 绘制 Reliability Diagrams
# reliability_diagrams(predict_output, label, n_bins=10)

