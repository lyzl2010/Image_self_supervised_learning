图像拼图算法实现https://arxiv.org/abs/1603.09246

permutations_1000.npy 是预设定的1000种图像打乱的方式

data_imagenet.py 针对imagenet文件夹格式数据实现图像的读取，把图像切分为9块，按照预定的1000种方式中的一种打乱

network.py 主网络，前期卷积层类似Alexnet网络，顺序提取9块的图像特征，然后concatenate起来输入全连接和分类层进行分类

train_imagenet.py 网络训练程序

实际效果：网络可以收敛，但是聚类实验后发现学不到太多语义信息。




