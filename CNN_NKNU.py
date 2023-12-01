from convolution.convByOOP import conv
from convolution.conventionalConvolutionOOP import ConventionalConv
from convolution.fixedKernel_conventionalConvolutionOOP import FixedKernelConventionalConv
from fullyconnected.createweight import createWeights
from fullyconnected.fullyconnectedOOP import fullyConnected
from PIL import Image

def img_num(a):
    l = len(a)
    s = ''
    for i in range(0, 4 - l):
        s += '0'
    s += a
    return s


if __name__ == '__main__':
    img_arr = []
    ans_arr = []
    # trainSize = 測試筆數, typeSize = 種類數目, total_size = trainSize x typeSize
    trainSize, typeSize, total_size = 100, 3, 300
    # leftRange = 權重、bias 左邊範圍, rightRange = 權重、bias 右邊範圍
    leftRange, rightRange = -0.1, 0.1
    path = ""
    for case in range(trainSize):
        for num in range(typeSize):

            # convly = conv('pic/{}/{}.png'.format(num,case), isMNIST, layers, times, strides)
            # handwritten number
            # convly = ConventionalConv('pic/{}/{}.png'.format(num, case), True, 2, 8, 1)
            # convly = conv('pic/{}/{}.png'.format(num,case), isMNIST, layers, times, strides)
            # CIFAR-10
            if (num == 0):
                path = 'CIFAR-10/train/airplane/{}.jpg'.format(img_num(str(case)))
            elif (num == 1):
                path = 'CIFAR-10/train/automobile/{}.jpg'.format(img_num(str(case)))
            else:
                path = 'CIFAR-10/train/bird/{}.jpg'.format(img_num(str(case)))
            convly = ConventionalConv(path, False, 2, 8, 1)
            # kernel set format : [convLayer][kernels in each layer][Y axis of the kernel][X axis of the kernel]
            kernelSet = convly.getKernelSet()
            # print(kernelSet)
            # print(len(convly.finalOutput))

            # conv_fixedKernal = FixedKernelConventionalConv(Path, kernelSet, isMNIST, layers, times per layer, strides)
            # kernel set format : [convLayer][kernels in each layer][Y axis of the kernel][X axis of the kernel]
            convly_fixedKernel = FixedKernelConventionalConv(path, kernelSet, False, 2, 8, 1)

            img_arr.append(convly.finalOutput)
            ans = convly.makeAnswer(num, typeSize)
            ans_arr.append(ans)

    input_size = len(convly.finalOutput)
    hidden_size = 300
    output_size = 3
    learning_rate = 0.001

    # 隨機初始化權重
    CW = createWeights(leftRange, rightRange, input_size, hidden_size, output_size, total_size)



    for epoch in range(5001):
        # 向前傳播
        fc = fullyConnected(img_arr, ans_arr, CW.weights_input_hidden, CW.bias_input_hidden,
                            CW.weights_hidden_output, CW.bias_hidden_output, "sigmoid", "MSE")
        fc.updateWeightSigmoid(fc.error, fc.hidden_layer_output, img_arr, CW.weights_hidden_output, fc.input_layer_hidden, CW.bias_hidden_output, CW.weights_input_hidden, CW.bias_input_hidden, learning_rate)
        trainAccuracy = CW.maxWeight(CW.weights_input_hidden,CW.weights_hidden_output,CW.bias_input_hidden,CW.bias_hidden_output,epoch,fc.hidden_layer_output,ans_arr)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, trainAccuracy: {trainAccuracy}%")
        if fc.loss <= 0.001:
            break
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {fc.loss}")

    # =======================================================================================以下測試
    img_arr2 = []
    ans_arr2 = []
    for case in range(trainSize):
        for num in range(typeSize):
            # convly = conv('pic/{}/{}.png'.format(num,case), isMNIST, layers, times, strides)
            # handwritten number
            # convly2 = ConventionalConv('pic/{}/{}.png'.format(num, case), True, 2, 8, 1)
            if (num == 0):
                path = 'CIFAR-10/valid/airplane/{}.jpg'.format(img_num(str(case)))
            elif (num == 1):
                path = 'CIFAR-10/valid/automobile/{}.jpg'.format(img_num(str(case)))
            else:
                path = 'CIFAR-10/valid/bird/{}.jpg'.format(img_num(str(case)))
            convly2 = ConventionalConv(path, False, 2, 8, 1)
            img_arr2.append(convly2.finalOutput)
            ans2 = convly2.makeAnswer(num, typeSize)
            ans_arr2.append(ans2)
    fc2 = fullyConnected(img_arr2, ans_arr2, CW.max_weights_input_hidden, CW.max_bias_input_hidden,
                        CW.max_weights_hidden_output, CW.max_bias_hidden_output, "sigmoid", "MSE")
    print("最終預測結果：")
    print(str(CW.acc(fc2.hidden_layer_output,ans_arr2)) + '%')
    # print(str(count) + '/' + str(len(predicted_output)))
    # print(str(count / len(predicted_output) * 100) + '%')
