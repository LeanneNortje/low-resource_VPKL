#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

from PIL import Image, ImageDraw
from torchvision import transforms

from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


TINT_COLOR = (100, 100, 100)  # Black
TRANSPARENCY = .25  # Degree of transparency, 0-100%
OPACITY = int(255 * TRANSPARENCY)

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

resize = transforms.Resize((100, 100))
to_tensor = transforms.ToTensor()
# image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

def myRandomCrop(im):

#         im = resize(im)
        im = to_tensor(im)
        return im

def load_image(impath):
    img = Image.open(impath).convert('RGB')
    # img = self.image_resize_and_crop(img)
    img = myRandomCrop(img)
#     img = image_normalize(img)
    return img.unsqueeze(0)

# fn = Path(f'dog_182805.jpg')
fns = list(Path('images').rglob(f'*.jpg'))
# chosen = np.random.choice(fns, size=2, replace=False)
# chosen = []
# for fn in fns:
#     print(fn)
#     a = input()
#     if a == 'y': chosen.append(fn)
#     elif a == 'x': break
data = {}
for fn in fns:
    c = Path(fn).stem.split('_')[0]
    if c not in data: data[c] = []
    data[c].append(fn)

data.keys()

model.to(0)
model.eval()
with torch.no_grad():
    for i, fn in enumerate(data['dog']):
    #     if i != 1: continue
        x = load_image(fn).to(0)
        predictions = model(x)
        image = Image.open(fn).convert('RGB')
        plt.figure()
        plt.imshow(image)
        plt.show()
        for j in range(predictions[0]['boxes'].size(0)):
            image = Image.open(fn).convert('RGB')
            w, h = image.size
            val = predictions[0]['boxes'][j, :].detach().cpu().numpy()
            x1 = val[0]
            x2 = val[2]
            y1 = val[1]
            y2 = val[3]
            im = image.crop((x1, y1, x2, y2))
            plt.imshow(im)
            plt.show()
            print('Choice: ')
            choice = input()
            if choice == 'y': key_image = im
            elif choice == 'x': break
        if choice == 'x': break


pooling = torchvision.ops.MultiScaleRoIAlign(['0', '1', '2', '3', 'pool'], 1, 2).to(0)

with torch.no_grad():
    key_features = model.backbone(to_tensor(key_image).unsqueeze(0).to(0))
    predictions = model(to_tensor(key_image).unsqueeze(0).to(0))
    key_boxes = []
    for p in range(predictions[0]['boxes'].size(0)): 
        key_boxes.append(predictions[0]['boxes'][p, :].unsqueeze(0))#.repeat(len(key_features), 1))
    sizes = [key_image.size]
    feat = pooling(key_features, key_boxes, sizes).detach().cpu().numpy()
#     key_features = key_features.detach().cpu().numpy()
#     predictions = predictions.detach().cpu().numpy()

    classes = ['cat', 'skateboard', 'women']
    dataset = {}
    for c in classes:
        for i, fn in tqdm(enumerate(data[c]), desc=f'{c}'):
            x = load_image(fn).to(0)
            features = model.backbone(x)
            predictions = model(x)

            boxes = []
            for p in range(predictions[0]['boxes'].size(0)): 
                boxes.append(predictions[0]['boxes'][p, :].unsqueeze(0))#.repeat(len(key_features), 1))
            sizes = [x.size]
            if c not in dataset: dataset[c] = []
            entry = pooling(features, boxes, sizes)[0, :, :, :].squeeze(-1).unsqueeze(0).detach().cpu().numpy()
            dataset[c].append(entry)
#             features = features.detach().cpu().numpy()
#             predictions = predictions.detach().cpu().numpy()
            break


# In[ ]:


feat.size()


# In[ ]:


for a in dataset:
    s = torch.bmm(feat[0, :, :, :].squeeze().unsqueeze(0).unsqueeze(0), a[0, :, :, :].squeeze(-1).unsqueeze(0)).squeeze().item()
    print(s)


# In[ ]:




