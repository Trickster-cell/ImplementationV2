from torchvision.datasets import Cityscapes #premade dataloader for cityscapes
import matplotlib.pyplot as plt
from PIL import Image
import torch 
from torch import nn

# %%
# dataset  = Cityscapes('../cityscapes/', split='train', mode='fine', target_type='semantic')

# %%
# fig, ax = plt.subplots(ncols=2, figsize=(12,8))
# ax[0].imshow(dataset[0][0])
# ax[1].imshow(dataset[0][1], cmap='gray')

# %%
ignore_index = 255
void_classes= [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30,-1]
valid_classes= [ignore_index, 7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
class_names = ['unlabeled',
               'road',
               'sidewalk',
               'building',
               'wall',
               'fence',
               'pole',
               'traffic light',
               'traffic sign',
               'vegetation',
               'terrain',
               'sky',
               'person',
               'rider',
               'car',
               'truck',
               'bus',
               'train',
               'motorcycle',
               'bicycle']
class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes = len(valid_classes)
class_map

# %%
colors =[
    [  0,   0,   0],
    [128,  64, 128],
    [244,  35, 232],
    [ 70,  70,  70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170,  30],
    [220, 220,   0],
    [107, 142,  35],
    [152, 251, 152],
    [ 70, 130, 180],
    [220,  20,  60],
    [255,   0,   0],
    [  0,   0, 142],
    [  0,   0,  70],
    [  0,  60, 100],
    [  0,  80, 100],
    [  0,   0, 230],
    [119,  11,  32],
]

label_colors = dict(zip(range(n_classes), colors))

# %%
def encode_segmap(mask):
    '''
    online mila tha
    remove unwanted classes and rectify the labels of wanted classes
    '''
    for void_c in void_classes:
        mask[mask == void_c] = ignore_index
    for valid_c in valid_classes:
        mask[mask == valid_c] = class_map[valid_c]

    return mask

# %%
def decode_segmap(temp):
    '''
    ye bhi online mila tha
    convert greyscale to color
    '''
    temp = temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colors[l][0]
        g[temp == l] = label_colors[l][1]
        b[temp == l] = label_colors[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = r/255.0
    rgb[:,:,1] = g/255.0
    rgb[:,:,2] = b/255.0

    return rgb

# %%
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision

class AdjustGamma:
    def __init__(self, gamma, gain=1):
        self.gamma = gamma
        self.gain = gain

    def __call__(self, image, mask):
        img = np.transpose(image,(2,0,1))
        gamma_tensor = torchvision.transforms.functional.adjust_gamma(torch.from_numpy(img), self.gamma, self.gain)
        img = np.transpose(gamma_tensor.numpy(), (1,2,0))
        return {'image': img, 'mask': mask}

transform = A.Compose(
    [
        A.augmentations.crops.transforms.RandomCrop (1024, 1024, always_apply=False),
        A.augmentations.geometric.rotate.SafeRotate (always_apply=False),
        A.Resize(256,256),
        A.HorizontalFlip(p=0.5),
        AdjustGamma(gamma=0.75),
        A.Normalize(mean = (0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225), max_pixel_value = float(225)),
        ToTensorV2(),
    ]
)

# %%
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.datasets import Cityscapes

class data_transform(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i,t in enumerate(self.target_type):
            if t == 'polygon':
                target = self.load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]


        if self.transforms is not None :
            transformed=transform(image=np.array(image), mask=np.array(target))
            return transformed['image'], transformed['mask']
        return image, target

# %%
dataset = data_transform('../cityscapes/', split='val', mode='fine', target_type='semantic', transforms=transform)
# img, seg = dataset[20]
# print(img.shape, seg.shape)