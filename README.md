# **Why does Knowledge Distillation Work? Rethink its Attention and Fidelity Mechanism**

This repository is the official PyTorch implementation of the paper.

## Requirement
```
pip install -r requirements.txt
```
## Data Preparation

  - Please download the [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html). After compressing the CIFAR-100 files, please run the following command to process the CIFAR dataset.
  ```
python dataloader/dealWith_cifar_dataset.py
  ```

  - For ImageNet_LT preparation,  we follow the instruction from this [link](https://github.com/zhmiao/OpenLongTailRecognition-OLTR)

  - Please download the [ImageNet_2012](http://image-net.org/index).



### Pretrained models

* Before training the teacher model, make sure to place the weights of ResNet50 in the `./resnet50.pt` directory. You can also use the following statement to download the weights.
```
# Download the weights for ResNet50.
model = models.resnet50(pretrained=True)

# Save the weights to a file.
torch.save(model.state_dict(), 'resnet50.pt')
```

## Training
### (1) CIFAR100
Train the teacher model.
```
bash script/cifar100-T.sh
```
Train the student model.
```
# 1T & 2T
bash script/cifar100.sh
# 3T
bash script/3T_cifar100.sh
```

### (2) CIFAR100_imb100
Train the teacher model.
```
bash script/cifar100_imb-T.sh
```
Train the student model.
```
# 1T & 2T
bash script/cifar100_imb.sh
# 3T
bash script/3T_cifar100_imb.sh
```

### (3) ImageNet
When using the balanced ImageNet dataset, make the following modifications to `DatasetFolder` in `torchvision/datasets/folder.py`.
```
#  Before modification:
  if self.transform is not None:
      sample = self.transform(sample)
  return sample, target
  
# After modification:
  if self.transform is not None:
      img_weak = self.transform[0](sample)
      img_strong = self.transform[1](sample)
  return img_weak, img_strong, target
```

Train the teacher model.
```
bash script/ImageNet-T.sh
```
Train the student model.
```
bash script/ImageNet.sh
```

### (4) ImageNet_LT
Train the teacher model.
```
bash script/ImageNet_LT-T.sh
```
Train the student model.
```
bash script/ImageNet_LT.sh
```

### Metric computation
Calculate the Expected Calibration Error (ECE) value.
```
bash script/ECE.sh
bash script/ECE_3T.sh
```

Compute the Intersection over Union (IoU) value.
```
bash script/IoU_TT.sh
bash script/IoU_TT_3T.sh
```

Calculate the affinity value.
```
python Affinity/calculate_affinity.py
python Affinity/calculate_affinity_3T.py
```