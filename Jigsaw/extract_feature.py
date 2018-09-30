import argparse
import os
from data_loader import DataLoader
import torch
import torch.nn as nn
from network import Network
import numpy as np
import time
import torchvision.transforms as transforms
from PIL import Image
import json
import torch.nn.functional as F


def main():
    path = "../data/val2014"
    dirs = os.listdir( path )
    print('loading model.......')
    model = nn.DataParallel(Network()).cuda()
    model.load_state_dict(torch.load('../model/models/model-100-76.ckpt'))
    model.eval()
    modules = list(list(list(model.children())[0].children())[0])[:-2]
    
    model=nn.Sequential(*modules)
    
    f_avg=open('conv5_norelu_avg.txt','w')
    f_max=open('conv5_norelu_max.txt','w')
    for file in dirs:
        try:
            img = Image.open(path+'/'+file).convert('RGB')
            img = img.resize([224, 224], Image.LANCZOS)
            img = transforms.ToTensor()(img).unsqueeze_(0).cuda()
            feature = model(img)
            avgpool=F.avg_pool2d(feature,kernel_size=26)
            avgpool=avgpool.view(256)
            avgpool=F.normalize(avgpool,dim=0)
            avgpool=avgpool.data.cpu().numpy().tolist()
            avg_item=[file,avgpool]
            f_avg.write('%s\n' % json.dumps(avg_item))

            maxpool=F.max_pool2d(feature,kernel_size=26)
            maxpool=maxpool.view(256)
            maxpool=F.normalize(maxpool,dim=0)
            maxpool=maxpool.data.cpu().numpy().tolist()
            max_item=[file,maxpool]
            f_max.write('%s\n' % json.dumps(max_item))
        except:
            pass
    
   

if __name__ == '__main__':
    main()
