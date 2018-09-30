# Visual-semantic-embdding
学习图像和文本的双向映射，利用metric learning的思想学习文本和图像的相似度模型。https://arxiv.org/abs/1707.05612


## 文件说明
* build_vocab.py 构建词汇数据集，英文一般用nltk分词，中文用jieba,分词后将词汇和index的映射存入两个dict中
* model_chinese.py 模型文件，利用cnn提取图像特征，rnn提取句子特征，计算相似度后利用ContrastiveLoss训练模型
* data_loader.py load数据文件，每条数据包含图像和对应的文本
* train_chinese.py 训练模型文件
## 模型效果
* 在AI challenger数据集上，可以实现以图搜词和以词搜图功能，且模型能够兼顾多个词汇的约束，学习到多个属性的含义
* 在噪声较多的数据集上效果未知
