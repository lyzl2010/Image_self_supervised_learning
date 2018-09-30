import numpy as np
import json

f=open('conv4_norelu_max.txt','r')
names=[]
feas=[]
i=0
for line in f:
	name,feature=json.loads(line)
	names.append(name)
	feas.append(np.asarray(feature))

	i+=1
	if i%100==0:
            print(i)
max=0
size=len(names)
np.save('conv4_norelu_max_names.npy',arr=np.asarray(names))
arr=np.zeros((size,size))
for i in range(size):
    if i%100==0:
        print(i)
    for j in range(i+1,size):
        #a=feas[i]/np.max(feas[i])
        #b=feas[j]/np.max(feas[j])
        arr[i][j]=arr[j][i]=np.mean(feas[i]*feas[j])
np.save('conv4_norelu_max_sim.npy',arr)
