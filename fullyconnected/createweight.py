import numpy as np
import random

class createWeights():

    weights_input_hidden, weights_hidden_output, bias_input_hidden, bias_hidden_output, max_weights_input_hidden, max_weights_hidden_output, max_bias_input_hidden, max_bias_hidden_output = [], [], [], [], [], [], [], []
    accuracy = 0

    def __init__(self, leftRange, rightRange, input_size, hidden_size, output_size, total_size):
        self.weights_input_hidden, self.weights_hidden_output = self.firstMakeWeight(leftRange, rightRange, input_size, hidden_size, output_size)
        self.bias_input_hidden, self.bias_hidden_output = self.firstMakeBias(leftRange, rightRange, total_size, hidden_size, output_size)
        self.max_weights_input_hidden, self.max_weights_hidden_output = self.firstMakeWeight(leftRange, rightRange, input_size, hidden_size, output_size)
        self.max_bias_input_hidden, self.max_bias_hidden_output = self.firstMakeBias(leftRange, rightRange, total_size, hidden_size, output_size)

    # 隨機初始化權重和bias
    def createWeights(self, leftRange, rightRange, input_size, output_size):
        np.random.seed(0)
        weights = np.random.uniform(leftRange, rightRange, size=(input_size, output_size))
        return weights

    # 第一次建立權重
    def firstMakeWeight(self, leftRange, rightRange, input_size, hidden_size, output_size):
        weights_input_hidden = self.createWeights(leftRange, rightRange, input_size, hidden_size)
        weights_hidden_output = self.createWeights(leftRange, rightRange, hidden_size, output_size)
        return weights_input_hidden, weights_hidden_output

    # 第一次建立bias
    def firstMakeBias(self, leftRange, rightRange, total_size, hidden_size, output_size):
        # total_size是(trainEnd-trainStart)*(typeEnd-typeStart)
        bias_input_hidden = self.createWeights(leftRange, rightRange, total_size, hidden_size)
        bias_hidden_output = self.createWeights(leftRange, rightRange, total_size, output_size)
        return bias_input_hidden, bias_hidden_output

    # 計算正確率
    def acc(self, predicted_output, ans):
        count = 0
        for i in range(len(predicted_output)):
            max1 = max(predicted_output[i])
            for j in range(len(predicted_output[i])):
                if (max1 == predicted_output[i][j]):
                    predicted_output[i][j] = 1
                else:
                    predicted_output[i][j] = 0
            # print("pre")
            # print(predicted_output[i])
            # print("ans")
            # print(ans[i])
            if ((predicted_output[i] == ans[i]).all()):
                count += 1
        return count / len(predicted_output) * 100

    # 紀錄最好的權重
    def maxWeight(self, weights_input_hidden, weights_hidden_output, bias_input_hidden, bias_hidden_output,
                  epoch, hidden_layer_output, ans):
        # global max_weights_input_hidden, max_weights_hidden_output, max_bias_input_hidden, max_bias_hidden_output
        if (epoch == 0 or self.acc(hidden_layer_output,ans) >= self.accuracy):
            self.max_weights_input_hidden, self.max_weights_hidden_output, self.max_bias_input_hidden, self.max_bias_hidden_output = weights_input_hidden, weights_hidden_output, bias_input_hidden, bias_hidden_output
            self.accuracy = self.acc(hidden_layer_output,ans)
        return self.accuracy