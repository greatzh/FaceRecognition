# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import random

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import numpy as np
from sklearn.model_selection import  train_test_split
 
from keras import backend as K
 
from data_all_prepreocessed import load_dataset, resize_image, IMAGE_SIZE
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
 
logs_test_dir = 'E:\python3\StarFaceRecognition\log'  # logs存储路径
 
class Dataset:
    def __init__(self, path_name):
        #训练集
        self.train_images = None
        self.train_labels = None
        
        
        #测试集
        self.test_images  = None            
        self.test_labels  = None
        
        #数据集加载路径
        self.path_name    = path_name
        
        #当前库采用的维度顺序
        self.input_shape = None
 
        self.nb_classes=None
 
        
    #加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, 
             img_channels = 1, nb_classes = 5): #灰度图 所以通道数为1 5个类别 所以分组数为5
        #加载数据集到内存
        images, labels = load_dataset(self.path_name)        
        
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = 0.3, random_state = random.randint(0, 100))   #将总数据按0.3比重随机分配给训练集和测试集    
        
 
        train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels) #由于TensorFlow需要通道数，我们上一步设置为灰度图，所以这里为1，否则彩色图为3
        test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
        self.input_shape = (img_rows, img_cols, img_channels)            
        
        #输出训练集、测试集的数量
        print(train_images.shape[0], 'train samples')
        print(test_images.shape[0], 'test samples')
                        
    
        #像素数据浮点化以便归一化
        train_images = train_images.astype('float32')            
        test_images = test_images.astype('float32')
        
        #将其归一化,图像的各像素值归一化到0~1区间
        train_images /= 255
        test_images /= 255
 
 
 
        self.train_images = train_images
        self.test_images  = test_images
        self.train_labels = train_labels
        self.test_labels  = test_labels
        self.nb_classes   = nb_classes
#最大池化 
#keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last') 
#一维卷积
#keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)    
 
 #建立CNN模型
class CNN(tf.keras.Model):
    #模型初始化
    def __init__(self):
        super().__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[3, 3],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu,   # 激活函数
        )
 
        self.conv3=tf.keras.layers.Conv2D( filters=32, kernel_size=[3, 3],  activation=tf.nn.relu )
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2])
        
        self.conv4=tf.keras.layers.Conv2D( filters=64, kernel_size=[3, 3], padding='same',  activation=tf.nn.relu )
        self.conv5=tf.keras.layers.Conv2D( filters=64, kernel_size=[3, 3],  activation=tf.nn.relu )
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2])
        
        self.flaten1=tf.keras.layers.Flatten()
        self.dense3 = tf.keras.layers.Dense(units=512,activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(units=5) #最后分类 5个单位
        
        
    #模型输出
    def call(self, inputs):
        x = self.conv1(inputs)                  
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool4(x)
        x = self.flaten1(x)
        x = self.dense3(x)
        x = self.dense4(x)
        output = tf.nn.softmax(x)
        print(output)
        plot_model(self, to_file='model.png')
        return output 
 
 
    #识别人脸
    def face_predict(self, image):    
 
        image = resize_image(image)
        image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1))                    
        
        #浮点并归一化
        image = image.astype('float32')
        image /= 255
        
        #给出输入属于各个类别的概率
        result = self.predict(image)
        #print('result:', result[0])
        
               
 
        #返回类别预测结果
        return result[0] 
 
    

    
if __name__ == '__main__':
    
    try:
        image_summary = tf.image_summary
        scalar_summary = tf.scalar_summary
        histogram_summary = tf.histogram_summary
        merge_summary = tf.merge_summary
        SummaryWriter = tf.train.SummaryWriter
    except:
        image_summary = tf.summary.image
        scalar_summary = tf.summary.scalar
        histogram_summary = tf.summary.histogram
        merge_summary = tf.summary.merge
        SummaryWriter = tf.summary.FileWriter
  
    
    
    
    learning_rate = 0.001 #学习率
    batch=32    #batch数
    EPOCHS = 120  #学习轮数
    log_dir = "./logs/"  # 日志保存路径 

    dataset = Dataset('./out_data/')    #数据都保存在这个文件夹下
    dataset.load()
    
    model = CNN()#模型初始化
    # 构造一个Tensorboard类的对象
    #tbCallBack = TensorBoard(log_dir="./log")
#    model.compile(optimizer=tf.train.AdamOptimizer(),
#                  loss='sparse categorical crossentropy',
#                  metrics=['accuracy'])
    # 在fit 里面指定callbacks参数
    #history=model.fit(dataset.test_images, dataset.test_labels, batch_size=batch, epochs=EPOCHS, shuffle=True, verbose=2, validation_split=0.2,callbacks=[tbCallBack])
    #print("Here is Model:")
    #print(model)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) #选择优化器
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy() #选择损失函数
   
    train_loss = tf.keras.metrics.Mean(name='train_loss') #设置变量保存训练集的损失值
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')#设置变量保存训练集的准确值
    test_loss = tf.keras.metrics.Mean(name='test_loss')#设置变量保存测试集的损失值
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')#设置变量保存测试集的准确值
   
    #tf.summary.scalar('loss', test_loss)
    #tf.summary.scalar('accuracy', test_accuracy)
    # 这个是log汇总记录
    #summary_op = tf.summary.merge_all()
    # 产生一个会话
    #sess = tf.Session()
    # 产生一个writer来写log文件
    #train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    #test_writer = tf.summary.FileWriter(logs_test_dir, sess.graph)
    # 产生一个saver来存储训练好的模型
    #saver = tf.test.Saver()
    # 所有节点初始化
    #sess.run(tf.global_variables_initializer())
    # 队列监控
#    merged_summary_op = tf.summary.merge_all()
#    # 数据保存器的初始化
#    saver = tf.train.Saver()
#
#    with tf.Session() as sess:
#    with tf.name_scope("loss"):
#        test_loss = tf.keras.metrics.Mean(name='test_loss')#设置变量保存测试集的损失值
#        tf.summary.scalar('loss',test_loss)
    
#    with tf.name_scope("accuracy"):
#        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')#设置变量保存测试集的准确值
#        tf.summary.scalar('accuracy',test_accuracy)
#    merged = tf.summary.merge_all()

 
 
    @tf.function
    def train_step(images, labels):
      with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))#优化器更新数据
 
      train_loss(loss)#更新损失值
      train_accuracy(labels, predictions)#更新准确值
 
    @tf.function
    def test_step(images, labels):
      predictions = model(images)
      t_loss = loss_object(labels, predictions)
 
      test_loss(t_loss)
      test_accuracy(labels, predictions)
      
#    def plot_acc_loss(loss, acc):
#        host = host_subplot(111)  # row=1 col=1 first pic
#        plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
#        par1 = host.twinx()   # 共享x轴
#     
#        # set labels
#        host.set_xlabel("steps")
#        host.set_ylabel("test-loss")
#        par1.set_ylabel("test-accuracy")
#     
#        # plot curves
#        p1, = host.plot(range(len(loss)), loss, label="loss")
#        p2, = par1.plot(range(len(acc)), acc, label="accuracy")
#     
#        # set location of the legend,
#        # 1->rightup corner, 2->leftup corner, 3->leftdown corner
#        # 4->rightdown corner, 5->rightmid ...
#        host.legend(loc=5)
#     
#        # set label color
#        host.axis["left"].label.set_color(p1.get_color())
#        par1.axis["right"].label.set_color(p2.get_color())
#     
#        # set the range of x axis of host and y axis of par1
#        # host.set_xlim([-200, 5200])
#        # par1.set_ylim([-0.1, 1.1])
#     
#        plt.draw()
#        plt.savefig("1.svg", format="svg")
#        plt.show() 
    def plot_acc_loss(loss):
        host = host_subplot(111)  # row=1 col=1 first pic
        plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
        par1 = host.twinx()   # 共享x轴
     
        # set labels
        host.set_xlabel("epoch")
        par1.set_ylabel("train-loss")
     
        # plot curves
        p2, = par1.plot(range(len(loss)), loss, label="loss")
     
        # set location of the legend,
        # 1->rightup corner, 2->leftup corner, 3->leftdown corner
        # 4->rightdown corner, 5->rightmid ...
        host.legend(loc=5)
     
        # set label color
        par1.axis["right"].label.set_color(p2.get_color())
     
        # set the range of x axis of host and y axis of par1
        # host.set_xlim([-200, 5200])
        # par1.set_ylim([-0.1, 1.1])
     
        plt.draw()
        plt.savefig("lossline.svg", format="svg")
        plt.show() 
        
     
#    with tf.Session() as sess:
#    # 日志记录
#        init = tf.global_variables_initializer()
#        sess.run(init)
#        writer = tf.summary.FileWriter(log_dir,sess.graph)
# 
#        for epoch in range(EPOCHS):
#         
#          train_ds = tf.data.Dataset.from_tensor_slices((dataset.train_images, dataset.train_labels)).shuffle(300).batch(batch)
#          test_ds = tf.data.Dataset.from_tensor_slices((dataset.test_images, dataset.test_labels)).shuffle(300).batch(batch)
#        
#          for images, labels in train_ds:
#            train_step(images, labels)
#         
#          for test_images, test_labels in test_ds:
#            test_step(test_images, test_labels)
            
        
        
        
    #eval_loss_list = []
    eval_acc_list = []
    
    for epoch in range(EPOCHS):
 
      train_ds = tf.data.Dataset.from_tensor_slices((dataset.train_images, dataset.train_labels)).shuffle(300).batch(batch)
      test_ds = tf.data.Dataset.from_tensor_slices((dataset.test_images, dataset.test_labels)).shuffle(300).batch(batch)
    
      for images, labels in train_ds:
        train_step(images, labels)
 
      for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
 
 
      
      # save testint result...
      #eval_loss_list.append(test_loss.result())
      eval_acc_list.append(train_loss.result())
      
      # 将loss与accuracy保存以供tensorboard使用
#      tf.summary.scalar('loss', cross_entropy)
#      tf.summary.scalar('accuracy', accuracy)
        
 
 
      #summary_str = sess.run(summary_op)
      #test_writer.add_summary(summary_str, epoch) 
      
      template = 'Epoch {} \nTrain Loss:{:.2f},Train Accuracy:{:.2%}\nTest Loss :{:.2f},Test Accuracy :{:.2%}'
      print (template.format(epoch+1,train_loss.result(),train_accuracy.result(),test_loss.result(),test_accuracy.result()))    #打印
 
    
    model.save_weights('./model/face1') #保存权重模型 命名为face1
    plot_acc_loss(eval_acc_list) 

 
    
 
 
