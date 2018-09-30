# 图像上色算法实现
https://arxiv.org/abs/1603.08511

## 文件说明
* model.py 原始网络模型
* model_column_unet.py 改进的模型：[U-net](https://arxiv.org/abs/1505.04597)加上[hypercolumn](https://arxiv.org/abs/1411.5752)
* training_layers.py 将ab颜色空间encode到313个类别和decode回去的文件，由作者提供的文件修改实现，依赖于resources中的文件
* resources 提供两个文件：pts_in_hull.npy 提供ab颜色空间到313个类别的映射关系，prior_probs.npy 提供算loss时给不同类别颜色的权重
* train_column.py 训练模型程序：可以视模型的不同import对应的模型，数据提供和模型存取路径需要自己提供
* sample_imagenet.py 预测程序，用来给图像进行上色，输入l空间的程序预测ab并将其合并为彩色图像存储

## 模型效果
* 上色效果优于同类其他模型
* 自监督效果好于拼图模型，和Examplar cnn模型相当

