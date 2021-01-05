# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
from face_train_final import CNN
import cv2
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import dlib
detector = dlib.get_frontal_face_detector() #获取人脸分类器
 
if __name__ == '__main__':
        
    #加载模型
    model = CNN()
    model.load_weights('./model/face1')    #读取模型权重参数
              
    #框住人脸的矩形边框颜色       
    color = (255, 0, 0)
    
    #人脸识别分类器本地存储路径
    cascade_path ="D:\Program Files\Python38\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"    
    #"E:\python3\StarFaceRecognition\test_liu.jpg" 
    #"E:\python3\TensorFlow1\face_recognition\test_liu.jpg"
    user=input("请选择图片（G）还是摄像头（V）:")
    if user=="G":
        path=input("请在此处输入图片路径：")
        img = cv2.imread(path)
        plt.imshow(img)
        plt.show()
        #timg=Image.open(path)
        #timg.show()
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cascade_path)
        
        dets = detector(img, 1) 
        print("检测到: {}张人脸，已灰度化处理".format(len(dets)))
        #利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(img_grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect
                    
                #截取脸部图像提交给模型识别这是谁
                image = img[y - 10: y + h + 10, x - 10: x + w + 10]
                face_probe = model.face_predict(image)   #获得预测值
                cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同
                pilimg = Image.fromarray(img)
                print('识别结果：')
                print('我:{:.2%}'.format(face_probe[0]))
                print('刘畅:{:.2%}'.format(face_probe[1]))
                print('刘德华:{:.2%}'.format(face_probe[2]))
                print('吴彦祖:{:.2%}'.format(face_probe[3]))
                print('郑爽:{:.2%}'.format(face_probe[4]))
               

            
        #cv2.destroyAllWindows()
        
        
    else:
        #捕获指定摄像头的实时视频流
        cap = cv2.VideoCapture(0)
        
        
        #循环检测识别人脸
        while True:
            ret, frame = cap.read()   #读取一帧视频
            
            if ret is True:
                
                #图像灰化，降低计算复杂度
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                continue
            #使用人脸识别分类器，读入分类器
            cascade = cv2.CascadeClassifier(cascade_path)                
     
            #利用分类器识别出哪个区域为人脸
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
            if len(faceRects) > 0:                 
                for faceRect in faceRects: 
                    x, y, w, h = faceRect
                    
                    #截取脸部图像提交给模型识别这是谁
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    face_probe = model.face_predict(image)   #获得预测值
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同
                    pilimg = Image.fromarray(frame)
                    draw = ImageDraw.Draw(pilimg) # 图片上打印 出所有人的预测值
                    font = ImageFont.truetype("simkai.ttf", 20, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
                    draw.text((x+25,y-95), '我:{:.2%}'.format(face_probe[0]), (255, 0, 0), font=font)
                    draw.text((x+25,y-70), '刘畅:{:.2%}'.format(face_probe[1]), (255, 0, 0), font=font)
                    draw.text((x+25,y-45), '刘德华:{:.2%}'.format(face_probe[2]), (255, 0, 0), font=font)
                    draw.text((x+25,y-20), '吴彦祖:{:.2%}'.format(face_probe[3]), (255, 0, 0), font=font)
                    draw.text((x+25,y-120),'郑爽:{:.2%}'.format(face_probe[4]), (255, 0, 0), font=font)
                    frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
                    
            cv2.imshow("ShowTime", frame)
            
            #等待10毫秒看是否有按键输入
            k = cv2.waitKey(10)
            #如果输入q则退出循环
            if k & 0xFF == ord('q'):
                break
     
        #释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows()
 
 
