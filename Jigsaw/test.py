import argparse
import os
from data_loader import DataLoader
import torch
import torch.nn as nn
from network import Network
import numpy as np
import time

def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main(args):
    print('Evaluating network.......')
    data_set = DataLoader(args.image_dir)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size = args.batch_size, shuffle = True,
                                              num_workers = args.num_workers)
    model = nn.DataParallel(Network()).cuda()
    model.load_state_dict(torch.load('../model/models/model-100-76.ckpt'))
    accuracy = []
    model.eval()
    for i, (images, labels, _) in enumerate(data_loader):
        #try:
            images = images.cuda()

            outputs = model(images)
            outputs = outputs.cpu().data

            prec1, prec5 = compute_accuracy(outputs, labels, topk=(1, 5))
            accuracy.append(prec1[0])
            print(prec1)
        #except:
        #    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = '../model/models/', help = 'path for saving trained models')
    parser.add_argument('--image_dir', type = str, default = '../data/val2014', help = 'directory for resized images')
    parser.add_argument('--log_step', type = int, default = 10, help = 'step size for prining log info')
    parser.add_argument('--save_step', type = int, default = 1000, help = 'step size for saving trained models')

    # Model parameters
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--num_workers', type = int, default = 1)
    args = parser.parse_args()
    print(args)
    main(args)
