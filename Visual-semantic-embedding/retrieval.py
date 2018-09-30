import torch
from torchvision import transforms
from model import VSE, cosine_sim
import numpy as np
import pickle
import os
import argparse
from evaluation import i2t, t2i
from data import get_test_loader
from PIL import Image
from vocab import Vocabulary 
import nltk
from shutil import copyfile

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(image_path,transform=None):
    image = Image.open(image_path)
    image = transform(image)
    return image


def load_model():
    
    print('start loading model')
    model_path = "checkpoint.pth.tar"
    vocab_path = "vocab"
    checkpoint = torch.load(model_path)
    print('model load')

    print('start loading vocab')
    # load vocabulary used by the model
    with open(os.path.join(vocab_path,'coco_vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    opt = checkpoint['opt']
    print('start construct model')
    # construct model
    model = VSE(opt)


    # load model state
    model.load_state_dict(checkpoint['model'])
    
    print('start reading image')
    data_dir = "data/coco/images/val2014"
    dirs = os.listdir(data_dir)
    features = []
    names = []
    i=0
    for file in dirs:
        try:
            i+=1
            if i%500==0:
                print(i)
            image = load_image(data_dir+'/'+file, transform)
            image = image.unsqueeze(0).float().cuda()
            feature = model.img_enc(image)
            arr=feature[0].data.cpu().numpy()
            features.append(arr)
            names.append(file)
        except:
            pass
    
    np.save('val_emb_retrain.npy',np.asarray(features))
    np.save('names_retrain.npy',np.asarray(names))

def text2img():
    model_path = "model_best.pth.tar"
    vocab_path="vocab"
    vocab_search_path = "./"
    checkpoint = torch.load(model_path)

    print('checkpoint loaded')
    with open(os.path.join(vocab_path, 'coco_vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    with open('coco_search_vocab.pkl', 'rb') as f:
        vocab_search = pickle.load(f)
    print('vocab loaded')
    vocab_size = len(vocab)
    opt = checkpoint['opt']
    
    model = VSE(opt)
    model.load_state_dict(checkpoint['model'])
    print('model loaded')
    ims=np.load('val_emb.npy')
    print('img emb loaded')
    name=np.load('names.npy')
    i=0
    for token in vocab_search.word2idx:
        try:
            #tokens="dog"
            if i%100==0:
                print(i)
            i+=1
            print('start '+token)
            os.mkdir('/mnt/cephfs/lab/wangyuqing/text_retrieval_select/'+token)
            tokens = nltk.tokenize.word_tokenize(str(token).lower())
            caption=[]
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            caption = torch.Tensor(caption).long().unsqueeze(0).cuda()
            cap_emb = model.txt_enc(caption,[caption.size()[1]])
            score=cosine_sim(torch.Tensor(ims).cuda(),cap_emb)
            score=score.data.cpu().numpy()
            ind=np.argsort(score,axis=0)[::-1][:20]
            for id in ind:
                img_name=name[id][0]
                copyfile('data/coco/images/val2014/'+img_name,'/mnt/cephfs/lab/wangyuqing/text_retrieval_select/'+token+'/'+img_name)
        except:
            pass
    
text2img()
#load_model()
