#------------------------------------------------
# Project: paddle
# Author:Peilin Wu - Najing Normal University
# File name :mnist-paddle.py
# Created time :2020/05
#------------------------------------------------
#加载飞桨和相关类库
#数据处理部分之前的代码，加入部分数据处理的库
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
import numpy as np
import os
import gzip
import json
import random

IMG_ROWS = 28
IMG_COLS = 28

#封装成函数
def load_data(mode = 'train'):
    datafile = './mnist.json.gz'
    print('从{}加载mnist数据集'.format(datafile))
    data = json.load(gzip.open(datafile))
    print("加载成功")

    #读取数据分为训练、验证、测试
    train_set,val_set,eval_set = data
    if mode == 'train':
        imgs,labels = train_set[0],train_set[1]
    elif mode == 'valid':
        # 获得验证数据集
        imgs, labels = val_set[0], val_set[1]
    elif mode == 'eval':
        # 获得测试数据集
        imgs, labels = eval_set[0], eval_set[1]
    else:
        raise Exception("mode can only be one of ['train', 'valid', 'eval']")
    print("数据集数量：",len(imgs))

    #校验数据
    imgs_lenth = len(imgs)
    assert len(imgs) == len(labels),\
        "数据数量{0},标签数量为{1}，数量不一致.".format(len(imgs), len(label))
    index_list = list(range(imgs_lenth))
    BATCHSIZE = 100

    #定义数据生成器
    def data_generator():
        if mode == 'train':#训练模式下需要打乱
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            #数据类型从list转换为folat32 ,shape为1，28，28
            img = np.reshape(imgs[i],[1,IMG_COLS,IMG_ROWS]).astype('float32')
            label = np.reshape(labels[i],[1]).astype('float32')
            imgs_list.append(img)
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list),np.array(labels_list)
                #清空buffer
                imgs_list = []
                labels_list = []

        if len(imgs_list) > 0: #"#如果遍历结束后，仍然有内容在buffer中
            yield np.array(imgs_list),np.array(labels_list)
    return data_generator

#定义神经网络
class MNIST(fluid.dygraph.Layer):
    def __init__(self,name_scope):
        super(MNIST,self).__init__(name_scope)
        #两个隐含层
        self.fc1 = Linear(input_dim=784,output_dim=10,act = 'sigmoid')
        self.fc2 = Linear(input_dim=10, output_dim=10, act='sigmoid')
        #输出层，不用激活函数
        self.fc3 = Linear(input_dim=10, output_dim=1, act=None)

    def forward(self,inputs,label = None):
        inputs = fluid.layers.reshape(inputs, [inputs.shape[0], 784])
        outputs1 = self.fc1(inputs)
        outputs2 = self.fc2(outputs1)
        outputs_final = self.fc3(outputs2)
        return outputs_final

# 网络结构部分之后的代码，保持不变
with fluid.dygraph.guard():
    model = MNIST("mnist")
    model.train()
    # 调用加载数据的函数，获得MNIST训练数据集
    train_loader = load_data('train')
    # 使用SGD优化器，learning_rate设置为0.01
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())
    # 训练5轮
    EPOCH_NUM = 5
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据
            image_data, label_data = data
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)

            # 前向计算的过程
            predict = model(image)

            # 计算损失，取一个批次样本损失的平均值
            loss = fluid.layers.square_error_cost(predict, label)
            avg_loss = fluid.layers.mean(loss)

            # 每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

    # 保存模型参数
    fluid.save_dygraph(model.state_dict(), 'mnist')