import torch
from torch.autograd import Variable
from skimage.color import lab2rgb
from model import Color_model
#from data_loader import ValImageFolder
import numpy as np
from skimage.color import rgb2lab, rgb2gray
import torch.nn as nn 
from PIL import Image
import scipy.misc
from torchvision import datasets, transforms
from training_layers import decode
import torch.nn.functional as F
import os

scale_transform = transforms.Compose([
    transforms.Scale(224),
    #transforms.CenterCrop(224),
    transforms.RandomCrop(224),
])

def load_image(image_path,transform=None):
    image = Image.open(image_path)
    
    if transform is not None:
        image = transform(image)
    #image_small=transforms.Scale(56)(image)
    #image_small=np.expand_dims(rgb2lab(image_small)[:,:,0],axis=-1)
    image=rgb2lab(image)[:,:,0]-50.
    image=torch.from_numpy(image).unsqueeze(0)
    return image

def main():
    data_dir = "/mnt/cephfs/lab/wangyuqing/imagenet_val"
    dirs=os.listdir(data_dir)
    color_model = nn.DataParallel(Color_model()).cuda().eval()
    color_model.load_state_dict(torch.load('../models/wpix/model-3-100.ckpt'))
     
    for file in dirs:
        try:
            image=load_image(data_dir+'/'+file, scale_transform)
            image=image.unsqueeze(0).float().cuda()
            img_ab_313=color_model(image)
            color_img=decode(image,img_ab_313)
            color_name = '/mnt/cephfs/lab/wangyuqing/wpix/' + file
            scipy.misc.imsave(color_name, color_img*255.)
        except:
            pass

if __name__ == '__main__':
    main()
