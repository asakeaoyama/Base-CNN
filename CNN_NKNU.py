from convolution.convByOOP import conv
from convolution.conventionalConvolutionOOP import ConventionalConv
from convolution.fixedKernel_conventionalConvolutionOOP import FixedKernelConventionalConv
from fullyconnected.createweight import createWeights
from fullyconnected.fullyconnectedOOP import fullyConnected
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
    # trainSize = 測試筆數, typeSize = 種類數目, total_size = trainSize x typeSize
    trainSize, typeSize, total_size = 100, 3, 300
    # leftRange = 權重、bias 左邊範圍, rightRange = 權重、bias 右邊範圍
    leftRange, rightRange = -0.1, 0.1
    for case in range(trainSize):
        for num in range(3,6):

            # convly = conv('pic/{}/{}.png'.format(num,case), True, 2, 8, 2)
            convly = ConventionalConv('pic/{}/{}.png'.format(num, case), True, 2, 8, 1)
            # kernel set format : [convLayer][kernels in each layer][Y axis of the kernel][X axis of the kernel]
            kernelSet = convly.getKernelSet()
            # print(kernelSet)
            # print(len(convly.finalOutput))

            # conv_fixedKernal = FixedKernelConventionalConv(Path, kernelSet, isMNIST, layers, times per layer, strides)
            # kernel set format : [convLayer][kernels in each layer][Y axis of the kernel][X axis of the kernel]
            convly_fixedKernel = FixedKernelConventionalConv('pic/{}/{}.png'.format(num, case), kernelSet, True, 2, 8,
                                                             1)

            img_arr.append(convly.finalOutput)
            ans = convly.makeAnswer(num, typeSize)
            ans_arr.append(ans)

    input_size = len(convly.finalOutput)
    hidden_size = 400
    output_size = 3
    learning_rate = 0.005

    # 隨機初始化權重
    CW = createWeights(leftRange, rightRange, input_size, hidden_size, output_size, total_size)



    for epoch in range(2001):
        # 向前傳播
        fc = fullyConnected(img_arr, ans_arr, CW.weights_input_hidden, CW.bias_input_hidden,
                            CW.weights_hidden_output, CW.bias_hidden_output, "sigmoid", "MSE")
        fc.updateWeightSigmoid(fc.error, fc.hidden_layer_output, img_arr, CW.weights_hidden_output, fc.input_layer_hidden, CW.bias_hidden_output, CW.weights_input_hidden, CW.bias_input_hidden, learning_rate)
        trainAccuracy = CW.maxWeight(CW.weights_input_hidden,CW.weights_hidden_output,CW.bias_input_hidden,CW.bias_hidden_output,epoch,fc.hidden_layer_output,ans_arr)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, trainAccuracy: {trainAccuracy}")
        if fc.loss <= 0.001:
            break
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {fc.loss}")

    # =======================================================================================以下測試
    img_arr2 = []
    ans_arr2 = []
    for case in range(trainSize):
        for num in range(3,6):

            # convly = conv('pic/{}/{}.png'.format(num, case), True, 2, 8, 2)
            convly2 = ConventionalConv('pic/{}/{}.png'.format(num, case), True, 2, 8, 1)
            img_arr2.append(convly2.finalOutput)
            ans2 = convly2.makeAnswer(num, typeSize)
            ans_arr2.append(ans2)
    fc2 = fullyConnected(img_arr2, ans_arr2, CW.max_weights_input_hidden, CW.max_bias_input_hidden,
                        CW.max_weights_hidden_output, CW.max_bias_hidden_output, "sigmoid", "MSE")
    print("最終預測結果：")
    print(str(CW.acc(fc2.hidden_layer_output,ans_arr2)) + '%')
    # print(str(count) + '/' + str(len(predicted_output)))
    # print(str(count / len(predicted_output) * 100) + '%')
