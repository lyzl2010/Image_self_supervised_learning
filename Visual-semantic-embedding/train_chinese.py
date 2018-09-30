import argparse
import torch
import torch.nn
import numpy as np
import os
import pickle
from model_chinese import VSE
from data_loader import get_loader
from build_vocab import Vocabulary
from torchvision import transforms

transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(args):
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
	                         transform, args.batch_size,
	                         shuffle = True, num_workers = args.num_workers)
    opt = parser.parse_args()
    opt.vocab_size = len(vocab)
    model = VSE(opt)
    model.load_state_dict(torch.load('models/1/25.ckpt'))
    total_step = len(data_loader)
    print(total_step)
    for epoch in range(args.num_epochs):
        try:
            for i,train_data in enumerate(data_loader):
                model.train_emb(*train_data)
                if i%10==0:
                    print('epoch: ',epoch, 'step: ',i)
            if epoch%5==0:    
                torch.save(model.state_dict(), os.path.join(args.model_path, '{}.ckpt'.format(epoch)))
        except:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = 'models/1/',
                        help = 'path for saving trained models')
    parser.add_argument('--vocab_path', type = str, default = './vocab.pkl',
                        help = 'path for vocabulary wrapper')
    parser.add_argument('--image_dir', type = str,
                        default = '/root/ai_challenger_zip/ai_challenger_caption_train_20170902',
                        help = 'directory for resized images')
    parser.add_argument('--caption_path', type=str, default='./coco_caption_train_annotations_20170902.json', help='path for train annotation json file')
    parser.add_argument('--margin', default = 0.2, type = float,
                        help = 'Rank loss margin.')
    parser.add_argument('--num_epochs', default = 300, type = int,
                        help = 'Number of training epochs.')
    parser.add_argument('--batch_size', default = 768, type = int,
                        help = 'Size of a training mini-batch.')
    parser.add_argument('--word_dim', default = 300, type = int,
                        help = 'Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default = 1024, type = int,
                        help = 'Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default = 2., type = float,
                        help = 'Gradient clipping threshold.')
    parser.add_argument('--num_layers', default = 1, type = int,
                        help = 'Number of GRU layers.')
    parser.add_argument('--learning_rate', default = .0001, type = float,
                        help = 'Initial learning rate.')
    parser.add_argument('--lr_update', default = 15, type = int,
                        help = 'Number of epochs to update the learning rate.')
    parser.add_argument('--num_workers', default = 10, type = int,
                        help = 'Number of data loader workers.')
    parser.add_argument('--log_step', default = 10, type = int,
                        help = 'Number of steps to print and record the log.')
    parser.add_argument('--max_violation', action = 'store_true',
                        help = 'Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default = 4096, type = int,
                        help = 'Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action = 'store_true',
                        help = 'Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default = 'vgg19',
                        help = """The CNN used for image encoder
                            (e.g. vgg19, resnet152)""")
    parser.add_argument('--measure', default = 'cosine',
                        help = 'Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action = 'store_true',
                        help = 'Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action = 'store_true',
                        help = 'Do not normalize the image embeddings.')
    args = parser.parse_args()
    print(args)
    main(args)
