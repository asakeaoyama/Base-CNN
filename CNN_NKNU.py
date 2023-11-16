from convolution.convByOOP import conv
from convolution.conventionalConvolutionOOP import ConventionalConv
from convolution.fixedKernel_conventionalConvolutionOOP import FixedKernelConventionalConv
from PIL import Image
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定義sigmoid的導数
def sigmoid_derivative(x):
    return x * (1 - x)


# 定義損失函數
def MSE(answer, predict):  # y=predict x=answer
    length = len(predict)
    lossSum = 0
    for i in range(length):
        lossSum += (predict - answer) ** 2
    lossSum *= (1 / 2)
    print('*******************')
    print('現在是Loss:' + str(lossSum))
    return lossSum / length


# 將RGB二值化
def flaten(dst):
    im = Image.open(dst, 'r')
    width, height = im.size
    pixel = list(im.getdata())
    arr = []
    for i in range(0, len(pixel)):
        arr.append(int(pixel[i][3]))
    return arr


if __name__ == '__main__':
    img_arr = []
    ans_arr = []
    for case in range(10, 60):
        for num in range(3, 6):

            # convly = conv('pic/{}/{}.png'.format(num,case), True, 2, 8, 2)
            convly = ConventionalConv('pic/{}/{}.png'.format(num, case), True, 2, 8, 1)
            # kernel set format : [convLayer][kernels in each layer][Y axis of the kernel][X axis of the kernel]
            kernelSet = convly.getKernelSet()
            print(kernelSet)
            print(len(convly.finalOutput))

            # conv_fixedKernal = FixedKernelConventionalConv(Path, kernelSet, isMNIST, layers, times per layer, strides)
            # kernel set format : [convLayer][kernels in each layer][Y axis of the kernel][X axis of the kernel]
            convly_fixedKernel = FixedKernelConventionalConv('pic/{}/{}.png'.format(num, case), kernelSet, True, 2, 8,
                                                             1)

            img_arr.append(convly.finalOutput)
            if num == 3:
                ans_arr.append([1, 0, 0])
            elif num == 4:
                ans_arr.append([0, 1, 0])
            else:
                ans_arr.append([0, 0, 1])

    input_size = len(convly.finalOutput)
    hidden_size = 400
    output_size = 3
    learning_rate = 0.005

    # 隨機初始化權重
    # np.random.seed(0)
    weights_input_hidden = np.random.uniform(-0.1, 0.1, size=(input_size, hidden_size))
    weights_hidden_output = np.random.uniform(-0.1, 0.1, size=(hidden_size, output_size))

    for epoch in range(200001):
        # 向前傳播
        input_layer_hidden = np.dot(img_arr, weights_input_hidden)
        input_layer_hidden = sigmoid(input_layer_hidden)
        hidden_layer_output = np.dot(input_layer_hidden, weights_hidden_output)
        hidden_layer_output = sigmoid(hidden_layer_output)

        # 計算損失（MSE）
        error = ans_arr - hidden_layer_output
        loss = np.mean(error ** 2)

        # 反向傳播
        d_output = error * sigmoid_derivative(hidden_layer_output)
        error_hidden = d_output.dot(weights_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(input_layer_hidden)

        # 更新權重
        weights_hidden_output += input_layer_hidden.T.dot(d_output) * learning_rate
        img_arr = np.array(img_arr)

        weights_input_hidden += img_arr.T.dot(d_hidden) * learning_rate
        if loss <= 0.001:
            break
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    # =======================================================================================以下測試
    img_arr2 = []
    for case in range(140, 190):
        for num in range(3, 6):

            # convly = conv('pic/{}/{}.png'.format(num, case), True, 2, 8, 2)
            convly = ConventionalConv('pic/{}/{}.png'.format(num, case), True, 2, 8, 1)
            img_arr2.append(convly.finalOutput)
    predicted_output = sigmoid(np.dot(sigmoid(np.dot(img_arr2, weights_input_hidden)), weights_hidden_output))

    ans = [0, 0, 0]
    count = 0
    for i in range(0, len(predicted_output)):
        if i % 3 == 0:
            ans = [1, 0, 0]
        elif i % 3 == 1:
            ans = [0, 1, 0]
        else:
            ans = [0, 0, 1]
        mm = max(predicted_output[i][0], predicted_output[i][1], predicted_output[i][2])
        arr = []
        for k in range(0, 3):
            if predicted_output[i][k] == mm:
                arr.append(1)
            else:
                arr.append(0)
        if arr == ans:
            count += 1

    print(str(count) + '/' + str(len(predicted_output)))
    print(str(count / len(predicted_output) * 100) + '%')
