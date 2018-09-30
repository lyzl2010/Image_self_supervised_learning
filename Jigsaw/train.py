import argparse
import os
from data_loader import DataLoader
import torch
import torch.nn as nn
from network import Network
import numpy as np
import time

def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    data_set = DataLoader(args.image_dir)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size = args.batch_size, shuffle = True,	                                          num_workers = args.num_workers)
    model=nn.DataParallel(Network()).cuda()
    model.load_state_dict(torch.load('../model/models/model-100-76.ckpt'))
    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr = args.learning_rate)
    total_step = len(data_loader)
    last_time=0
    for epoch in range(args.num_epochs):
        for i, (images, targets, original) in enumerate(data_loader):
            images=images.cuda()
            targets=targets.cuda()
            outputs = model(images)
            #print('outputs',outputs[0])
            loss = criterion(outputs, targets)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            # Print log info
            this_time=time.time()-last_time
            last_time=time.time()
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.4f}'.format(epoch, args.num_epochs, i, total_step, loss.item(),this_time))
	    # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                torch.save(model.state_dict(), os.path.join(args.model_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = '../model/models/', help = 'path for saving trained models')
    parser.add_argument('--image_dir', type = str, default = '../data/images256', help = 'directory for resized images')
    parser.add_argument('--log_step', type = int, default = 1, help = 'step size for prining log info')
    parser.add_argument('--save_step', type = int, default = 76, help = 'step size for saving trained models')

    # Model parameters
    parser.add_argument('--num_epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 1024)
    parser.add_argument('--num_workers', type = int, default = 2)
    parser.add_argument('--learning_rate', type = float, default = 1e-5)
    args = parser.parse_args()
    print(args)
    main(args)
