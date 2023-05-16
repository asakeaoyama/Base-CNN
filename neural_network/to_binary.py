# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:26:12 2023

@author: Ming Yao
"""
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt

def to_bin(dst):
    im = Image.open(dst, 'r')
    width, height = im.size
    pixel = list(im.getdata())
    print(pixel)
    arr = []
    for i in range(0,len(pixel)):
        count = 0
        count+=pixel[i][3]
        if(pixel[i][3] >150):
            arr.append(int(1))
        else:
            arr.append(int(0))
    return arr

img = Image.new('RGB',(28,28),'white')
drawobj  = ImageDraw.Draw(img)
five = to_bin("D:/python_code/numbers/dataset/2/2/10.png")
count = 0
for i in range(0,28):
    for j in range(0,28):
        if(five[count] == 1):
            drawobj.point([j,i],'black')
        else:
            drawobj.point([j,i],'white')
        count+=1
plt.imshow(img)

# =============================================================================
# 
# b = np.array([one,two,three,four,five,six,seven,eight,nine,test2,test7,test8,test6,test3,test4,test1,test5,test9])
# c = np.array([ans(1),ans(2),ans(3),ans(4),ans(5),ans(6),ans(7),ans(8),ans(9),ans(2),ans(7),ans(8),ans(6),ans(3),ans(4),ans(1),ans(5),ans(9)])
# =============================================================================

# =============================================================================
# import numpy as np
# from PIL import Image
# def sigmoid(x):#activative function
#     return 1 / (1 + np.exp(-x))
# 
# def sigmoid_derivative(x):#activative function
#     return x*(1-x)
# 
# def to_bin(dst):
#     im = Image.open(dst, 'r')
#     width, height = im.size
#     pixel = list(im.getdata())
#     arr = []
#     for i in range(0,len(pixel)):
#         count = 0
#         for rgb in range(0,3):
#             count+=pixel[i][rgb]
#         if(count <470):
#             arr.append(int(1))
#         else:
#             arr.append(int(0))
#     return arr
# 
# one = to_bin("C:/Users/user/Desktop/img/1.png")
# two = to_bin("C:/Users/user/Desktop/img/2.png")
# three = to_bin("C:/Users/user/Desktop/img/3.png")
# four = to_bin("C:/Users/user/Desktop/img/4.png")
# five = to_bin("C:/Users/user/Desktop/img/5.png")
# 
# test1 = to_bin("C:/Users/user/Desktop/img/1_test.png")
# test2 = to_bin("C:/Users/user/Desktop/img/2_test.png")
# test3 = to_bin("C:/Users/user/Desktop/img/3_test.png")
# test4 = to_bin("C:/Users/user/Desktop/img/4_test.png")
# test5 = to_bin("C:/Users/user/Desktop/img/5_test.png")
# 
# oneone = to_bin("C:/Users/user/Desktop/img/1_1.png")
# twotwo = to_bin("C:/Users/user/Desktop/img/2_1.png")
# threethree = to_bin("C:/Users/user/Desktop/img/3_1.png")
# fourfour = to_bin("C:/Users/user/Desktop/img/4_1.png")
# fivefive = to_bin("C:/Users/user/Desktop/img/5_1.png")
# 
# one1 = to_bin("C:/Users/user/Desktop/img/1_1.png")
# two1 = to_bin("C:/Users/user/Desktop/img/2_2.png")
# three1 = to_bin("C:/Users/user/Desktop/img/3_2.png")
# four1 = to_bin("C:/Users/user/Desktop/img/4_2.png")
# five1 = to_bin("C:/Users/user/Desktop/img/5_2.png")
# 
# four2 = to_bin("C:/Users/user/Desktop/img/4_3.png")
# five2 = to_bin("C:/Users/user/Desktop/img/5_3.png")
# 
# four3 = to_bin("C:/Users/user/Desktop/img/4_4.png")
# five3 = to_bin("C:/Users/user/Desktop/img/5_4.png")
# 
# four4 = to_bin("C:/Users/user/Desktop/img/4_5.png")
# five4 = to_bin("C:/Users/user/Desktop/img/5_5.png")
# 
# four5 = to_bin("C:/Users/user/Desktop/img/4_6.png")
# five5 = to_bin("C:/Users/user/Desktop/img/5_6.png")
# 
# four6 = to_bin("C:/Users/user/Desktop/img/4_7.png")
# five6 = to_bin("C:/Users/user/Desktop/img/5_7.png")
# 
# four7 = to_bin("C:/Users/user/Desktop/img/4_8.png")
# five7 = to_bin("C:/Users/user/Desktop/img/5_8.png")
# 
# four8 = to_bin("C:/Users/user/Desktop/img/4_9.png")
# five8 = to_bin("C:/Users/user/Desktop/img/5_9.png")
# 
# def ans(number):
#     arr = []
#     for i in range(1,3):
#         if(i == number):
#             arr.append(1)
#         else:
#             arr.append(0)
#     return arr
# class BPnn:
#     def __init__(self, input_matrix, output_matrix, layer_size, learn_rate):
#         self.layers = [] #每一層的值
#         self.input = input_matrix
#         #print(type(self.input))
#         self.output = output_matrix
#         self.layer_neural_count = len(layer_size) - 1 #神經層數
#         self.layer_size = layer_size
#         self.weight = {} #權重
#         self.bias = {} #掰厄斯
#         self.learn_rate = learn_rate #學習率
#         for i in range(1, self.layer_neural_count+1):
#             self.weight[i-1] = np.random.rand(self.layer_size[i-1], self.layer_size[i])
#             self.bias[i-1] = np.random.rand(1, self.layer_size[i])
# 
#     def forward(self, _input): #向前傳求各層的輸出
#         self.layers = []    #每次把保存的值清空
#         self.layers.append(_input) 
#         for i in range(self.layer_neural_count):
#             raw_value = np.dot(self.layers[i], self.weight[i]) + self.bias[i]   #求各層各神經元的RAW值
#             active_value = sigmoid(raw_value)   #用激活函数進行變換
#             self.layers.append(active_value)    #將每層求得的值保存，每次新求的值利用上層的值
# 
#     def back_propagation(self, _output):
#         update_list = []
#         theta_output = (self.layers[-1] - _output) * sigmoid_derivative(self.layers[-1])  #用誤差平方和的一半對輸出求導
#         update_list.append(theta_output)
#         for i in range(self.layer_neural_count-1, 0, -1):#用公式對BP計算
#             theta_hidden = np.dot(update_list[-1], self.weight[i].T) * sigmoid_derivative(self.layers[i])
#             update_list.append(theta_hidden)
#         update_list.reverse()
#         change_weight = {}
#         change_bias = {}
# 
#         for i in range(len(update_list)):
#             change_weight[i] = np.dot(self.layers[i].T, update_list[i]) * self.learn_rate
#             change_bias[i] = update_list[i] * self.learn_rate
# 
#         for i in range(len(update_list)):       #更新權重和掰厄斯
#             self.weight[i] -= change_weight[i]
#             self.bias[i] -= change_bias[i]
# 
#         return (sum((self.layers[-1]-_output)**2)/2)
# 
#     def epoch(self, count):
#         j = 0 #迭代次數
#         for i in range(count):
#             for x, y in zip(self.input, self.output):
#                 j += 1
#                 self.forward(x.reshape(1, len(x))) 
#                 if self.back_propagation(y.reshape(1, len(y))).all() < 0.001: #誤差小於0.001直接完成勒
#                     print("第 "+str(j)+" 次誤差小於0.001")
#                     return
# 
#     def predict(self, input_array):
#         self.layers = []  # 每次把保存的值清空
#         self.layers.append(input_array)  # 輸入加進保存
#         for i in range(self.layer_neural_count):#層數
#             raw_value = np.dot(self.layers[i], self.weight[i]) + self.bias[i]#求各層各神經元的raw值
#             active_value = sigmoid(raw_value)  # 用激活函数進行變換
#             self.layers.append(active_value)
#         print(self.layers[-1])
#         
# 
# b = np.array([four,test4,fourfour,four1,four2,four3,four4,four5,four6,four7,four8,five,test5,fivefive,five1,five2,five3,five4,five5,five6,five7])
# c = np.array([ans(1),ans(1),ans(1),ans(1),ans(1),ans(1),ans(1),ans(1),ans(1),ans(1),ans(1),ans(2),ans(2),ans(2),ans(2),ans(2),ans(2),ans(2),ans(2),ans(2),ans(2)])
# a = BPnn(b, c, [100,20,15,2], 0.2)
# a.epoch(10000)
# test = to_bin("C:/Users/user/Desktop/img/none.png")
# a.predict(np.array([test]))
# =============================================================================
