import numpy as np
import json
from shutil import copyfile
import os

f=open('ind_conv5_norelu_avg.txt','r')
names=np.load('conv5_norelu_avg_names.npy')
marked=[0 for i in range(names.shape[0])]

i=0
for line in f:
    id_list=json.loads(line)
    print(len(id_list))
    
    if len(id_list)>10:
        if marked[i]==0:
            os.mkdir('../data/cluster/'+str(i))
            for id in id_list:
                copyfile('../data/val2014/'+names[id][0],'../data/cluster/'+str(i)+'/'+names[id][0])
    i+=1    
