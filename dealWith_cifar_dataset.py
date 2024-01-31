#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:10:32 2023

@author: ps
"""

# code dealing with the downloaded cifar-10 and cifar-100 dataset.

import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os



srcRootDir = '/mnt/d/data/cifar-100-python/'
# srcRootDir = '/mnt/d/data/cifar-10-batches-py/'
save_train = 'clean_img/train/'
save_val = 'clean_img/val/'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



if __name__ == '__main__':
    # for phase in ['test','train']:
    for phase in ['train']:
        # CIFAR-100
        fullDataFileName = srcRootDir + phase #'test_batch' 'test'
        # # cifar-10
        # fullDataFileName = srcRootDir + 'data_batch_1' #'train' #'data_batch_1-5'
        this_dict = unpickle(fullDataFileName)
        
        #key_data = bytes('data', encoding='utf-8')
        pixels = this_dict[b'data'] # array of images: size (10000, 3072) for CIFAR-10, or (50000, 3072) for CIFAR-100
        # labels = this_dict[b'labels'] # for CIFAR-10
        labels = this_dict[b'fine_labels'] # for CIFAR-100
        filenames = this_dict[b'filenames']
        n_images = len(labels)
        
        for idx in range(n_images):
            print(idx)
            
            this_img = pixels[idx,:]
            this_label = labels[idx] # int
            this_filename = str(filenames[idx], 'utf-8')
            #CIFAR-100
            # this_dstDir = srcRootDir + 'clean_img/val/' + str(this_label) + '/' # test
            if phase == 'test':
                this_dstDir = srcRootDir + save_val + str(this_label) + '/' # test
            elif phase == 'train':
                this_dstDir = srcRootDir + save_train + str(this_label) + '/' #train
            if not os.path.exists(this_dstDir):
                # os.mkdir(this_dstDir)
                os.makedirs(this_dstDir)
            
            this_img_R = this_img[:1024].reshape(32,32)
            this_img_G = this_img[1024:2048].reshape(32,32)
            this_img_B = this_img[2048:].reshape(32,32)
            
            this_img_RGB = np.dstack((this_img_R, this_img_G, this_img_B)) # size (32,32,3)
            
            #plt.imshow(this_img_RGB)
            #plt.show()
            
            im = Image.fromarray(this_img_RGB)
            im.save(this_dstDir + this_filename)
            
            #print()





