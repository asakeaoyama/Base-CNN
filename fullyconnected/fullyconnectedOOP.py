from PIL import Image
import numpy as np
import random

# 測試的全連接層
class fullyConnected:

    # 第一層神經元
    input_layer_hidden = []
    # 最終預測矩陣
    hidden_layer_output = []
    # 測試的損失值
    loss = 0
    # 測試的error值
    error = 0

    def __init__(self, X,Y,weights_input_hidden,bias_input_hidden,weights_hidden_output,bias_hidden_output,activation, lossFunction):
        # 全連接
        # 使用sigmoid
        if (activation == "sigmoid"):
            self.input_layer_hidden, self.loss, self.hidden_layer_output, self.error = self.fullyConnectSigmoid(X, Y, weights_input_hidden, bias_input_hidden, lossFunction, weights_hidden_output, bias_hidden_output)


    # 定義sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 定義sigmoid的導数
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # 定義relu
    def relu(self, x):
        return max(0, x)

    # 定義損失函數
    def MSE(self, answer, predict):  # y=predict x=answer
        # 計算損失（MSE）
        error = answer - predict
        loss = np.mean(error ** 2)
        return error, loss

    # 全連接層sigmoid
    def fullyConnectSigmoid(self, X, Y, weights_input_hidden, bias_input_hidden, lossFunction, weights_hidden_output, bias_hidden_output):
        # 向前傳播1
        input_layer_hidden = np.dot(X, weights_input_hidden)
        input_layer_hidden += bias_input_hidden
        input_layer_hidden = self.sigmoid(input_layer_hidden)
        hidden_layer_output = np.dot(input_layer_hidden, weights_hidden_output)
        hidden_layer_output += bias_hidden_output
        hidden_layer_output = self.sigmoid(hidden_layer_output)
        if (lossFunction == "MSE"):
            error, loss = self.MSE(Y, hidden_layer_output)
        return input_layer_hidden, loss, hidden_layer_output, error

    #更新權重
    def updateWeightSigmoid(self, error, hidden_layer_output, X, weights_hidden_output, input_layer_hidden, bias_hidden_output, weights_input_hidden, bias_input_hidden, learning_rate):
        # 反向傳播1
        d_output = error * self.sigmoid_derivative(hidden_layer_output)
        error_hidden = d_output.dot(weights_hidden_output.T)
        d_hidden = error_hidden * self.sigmoid_derivative(input_layer_hidden)
        # 更新權重1
        weights_hidden_output += input_layer_hidden.T.dot(d_output) * learning_rate
        bias_hidden_output += d_output * learning_rate
        X = np.array(X)  # transfer list to array
        weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        bias_input_hidden += d_hidden * learning_rate

