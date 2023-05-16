# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:26:12 2023

@author: Ming Yao
"""
import numpy as np
# b = np.random.rand(3, 2)
# print(b)
# d = np.array([[1]]*3).T
def sigmoid(x):#activative function
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):#activative function
    return x*(1-x)


class BPnn:
    def __init__(self, input_matrix, output_matrix, layer_size, learn_rate):
        self.layers = [] #每一層的值
        self.input = input_matrix
        #print(type(self.input))
        self.output = output_matrix
        self.layer_neural_count = len(layer_size) - 1 #神經層數
        self.layer_size = layer_size
        self.weight = {} #權重
        self.bias = {} #掰厄斯
        self.learn_rate = learn_rate #學習率
        for i in range(1, self.layer_neural_count+1):
            self.weight[i-1] = np.random.rand(self.layer_size[i-1], self.layer_size[i])
            self.bias[i-1] = np.random.rand(1, self.layer_size[i])

    def forward(self, _input): #向前傳求各層的輸出
        self.layers = []    #每次把保存的值清空
        self.layers.append(_input) 
        for i in range(self.layer_neural_count):
            raw_value = np.dot(self.layers[i], self.weight[i]) + self.bias[i]   #求各層各神經元的RAW值
            active_value = sigmoid(raw_value)   #用激活函数進行變換
            self.layers.append(active_value)    #將每層求得的值保存，每次新求的值利用上層的值

    def back_propagation(self, _output):
        update_list = []
        theta_output = (self.layers[-1] - _output) * sigmoid_derivative(self.layers[-1])  #用誤差平方和的一半對輸出求導
        update_list.append(theta_output)
        for i in range(self.layer_neural_count-1, 0, -1):#用公式對BP計算
            theta_hidden = np.dot(update_list[-1], self.weight[i].T) * sigmoid_derivative(self.layers[i])
            update_list.append(theta_hidden)
        update_list.reverse()
        change_weight = {}
        change_bias = {}

        for i in range(len(update_list)):
            change_weight[i] = np.dot(self.layers[i].T, update_list[i]) * self.learn_rate
            change_bias[i] = update_list[i] * self.learn_rate

        for i in range(len(update_list)):       #更新權重和掰厄斯
            self.weight[i] -= change_weight[i]
            self.bias[i] -= change_bias[i]

        return (sum((self.layers[-1]-_output)**2)/2)

    def epoch(self, count):
        j = 0 #迭代次數
        for i in range(count):
            for x, y in zip(self.input, self.output):
                j += 1
                self.forward(x.reshape(1, len(x))) 
                if self.back_propagation(y.reshape(1, len(y))).any() < 0.001: #誤差小於0.001直接完成勒
                    print("第 "+str(j)+" 次誤差小於0.001")
                    return

    def predict(self, input_array):
        self.layers = []  # 每次把保存的值清空
        self.layers.append(input_array)  # 輸入加進保存
        for i in range(self.layer_neural_count):#層數
            raw_value = np.dot(self.layers[i], self.weight[i]) + self.bias[i]#求各層各神經元的raw值
            active_value = sigmoid(raw_value)  # 用激活函数進行變換
            self.layers.append(active_value)
        print(self.layers[-1])
        
       
b = np.array([[1,1],[0,0.05],[0.2,0.1],[0.9,0.8],[0.5,0.5],[0.7,0.6],[0.3,0.35],[0.6,0.9]])#(降雨機率，AQI/100)
c = np.array([[0],[1],[1],[0],[1],[0],[1],[0]])#1適合,0不適合
a = BPnn(b, c, [2,3,5,1], 0.2)
a.epoch(2000)
a.predict(np.array([[0.9,0.9]]))