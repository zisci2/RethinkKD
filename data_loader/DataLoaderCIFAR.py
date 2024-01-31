import os
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Sampler

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.
        cifar100也使用
        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

class SubPolicy(object):

    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int64),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        func = {
            "shearX": self.shearX,
            "shearY": self.shearY,
            "translateX": self.translateX,
            "translateY": self.translateY,
            "rotate": self.rotate,
            "color": self.color,
            "posterize": self.posterize,
            "solarize": self.solarize,
            "contrast": self.contrast,
            "sharpness": self.sharpness,
            "brightness": self.brightness,
            "autocontrast": self.autocontrast,
            "equalize": self.equalize,
            "invert": self.invert
        }
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]
        self.fillcolor = fillcolor  ##

    # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
    def rotate_with_fill(self, img, magnitude):
        rot = img.convert("RGBA").rotate(magnitude)
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

    # 具体的增强方式
    def shearX(self, img, magnitude):
        return img.transform(
            img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)

    def shearY(self, img, magnitude):
        return img.transform(
            img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)

    def translateX(self, img, magnitude):
        return img.transform(
            img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
            fillcolor=self.fillcolor)

    def translateY(self, img, magnitude):
        return img.transform(
            img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
            fillcolor=self.fillcolor)

    def rotate(self, img, magnitude):
        return self.rotate_with_fill(img, magnitude)

    def color(self, img, magnitude):
        return ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1]))

    def posterize(self, img, magnitude):
        return ImageOps.posterize(img, magnitude)

    def solarize(self, img, magnitude):
        return ImageOps.solarize(img, magnitude)

    def contrast(self, img, magnitude):
        return ImageEnhance.Contrast(img).enhance(1 + magnitude * random.choice([-1, 1]))

    def sharpness(self, img, magnitude):
        return ImageEnhance.Sharpness(img).enhance(1 + magnitude * random.choice([-1, 1]))

    def brightness(self, img, magnitude):
        return ImageEnhance.Brightness(img).enhance(1 + magnitude * random.choice([-1, 1]))

    def autocontrast(self, img, magnitude):
        return ImageOps.autocontrast(img)

    def equalize(self, img, magnitude):
        return ImageOps.equalize(img)

    def invert(self, img, magnitude):
        return ImageOps.invert(img)



    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img



data_transforms_CIFAR = {

    'weak': transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010])
    ]),  
    'strong': transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3), 
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(), # add AutoAug
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16), 
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),#
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010])
    ])
}


class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        # if self.transform is not None:
        if self.transform is not None and len(self.transform) != 1:
            # img_base = self.transform[0](sample)
            img_weak = self.transform[0](sample)
            img_strong = self.transform[1](sample)
        if self.transform is not None and len(self.transform) == 1: 
            # img_base = self.transform[0](sample)
            img_weak = self.transform[0](sample)
            img_strong = 1
            
        return img_weak, img_strong, label



def Load_CIFAR100(data_root, dataset, phase, batch_size, num_workers=4, shuffle=True):
    txt = None
    if dataset == 'CIFAR100_imb100':
        if phase == 'train':
            txt = 'data_txt/cifar100/CIFAR100_train_imb100.txt'
        elif phase == 'val' or phase == 'val_aug': #or phase == 'test':
            txt = 'data_txt/cifar100/CIFAR100_val.txt'
            
    elif dataset == 'CIFAR100':
        if phase == 'train':
            txt = 'data_txt/cifar100/train_paths.txt'
        elif phase == 'val' or phase == 'val_aug': #or phase == 'test':
            txt = 'data_txt/cifar100/val_paths.txt'
    elif txt is None:
        assert False

    print('Loading data from %s' % (txt))



    if phase == 'train':
        # set_ = LT_Dataset(data_root, txt, transform=[data_transforms_CIFAR['base'],
        set_ = LT_Dataset(data_root, txt, transform=[
                                                    data_transforms_CIFAR['weak'],
                                                    data_transforms_CIFAR['strong'],
                                                ])
        print("The data transforms used are:\nstrong: {}\nweak: {}\n".format(
                                                                    # data_transforms_CIFAR['base'],
                                                                    data_transforms_CIFAR['weak'],
                                                                    data_transforms_CIFAR['strong'],
                                                                    ))
    elif phase == 'val':
        set_ = LT_Dataset(data_root, txt, transform=[data_transforms_CIFAR['test']])
        print("The data transform used is:\ntest:{}".format(data_transforms_CIFAR['test']))
    
    else: # newly added by Chenqi: for 'val_aug'
        set_ = LT_Dataset(data_root, txt, transform=[
                                                    data_transforms_CIFAR['weak'],
                                                    data_transforms_CIFAR['strong'],
                                                ])
        print("The data transforms used are:\nstrong: {}\nweak: {}\n".format(
                                                                    # data_transforms_CIFAR['base'],
                                                                    data_transforms_CIFAR['weak'],
                                                                    data_transforms_CIFAR['strong'],
                                                                    ))
        
    print("The dataset is loaded with {} samples, sourced from {}.".format(len(set_),txt))


    print('Shuffle is %s.' % (shuffle))
    
    if phase == 'val_aug':
        return DataLoader(dataset=set_, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers,
                              drop_last=True)
    else:
        return DataLoader(dataset=set_, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers,
                            #   drop_last=True
                              )


if __name__ == "__main__":
    data_root = '/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img' #"/mnt/d/data/cifar-100-python/clean_img"
    dataset = 'CIFAR100_imb100'
    phase = 'train'
    batch_size = 1
    aaaaa = Load_CIFAR100(data_root, dataset, phase, batch_size, num_workers=4, shuffle=True)
    for sample, img1, img2, label in aaaaa:
        print(sample)
        print()
        print(img1)
        print()
        print(img2)
        print()
        print(label)
        break