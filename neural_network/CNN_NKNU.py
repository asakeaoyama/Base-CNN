from PIL import Image
import numpy as np
import random
from math import ceil

width = 0
height = 0

def importImgToData(path):
    global height, width
    orgImg = Image.open(path)
    pxMatrix = np.asarray(orgImg)

    height = orgImg.height
    width = orgImg.width
    data = [[[0 for rgb in range(3)] for x in range(width)] for y in range(height)]
    #print(height, width)

    for y in range(height):
        for x in range(width):
            r, g, b = pxMatrix[y][x]
            #print(r, g, b)
            data[y][x][0] = r
            data[y][x][1] = g
            data[y][x][2] = b

    return data

def importMNISTToData(path):
    orgImg = Image.open(path)
    pxMatrix = np.asarray(orgImg)
    # print("pic dim :", pxMatrix.ndim)

    height = orgImg.height
    width = orgImg.width
    data = [[[0 for rgb in range(3)] for x in range(width)] for y in range(height)]
    # print(height, width)

    for y in range(height):
        for x in range(width):
            r, g, b, a= pxMatrix[y][x]
            # print(r, g, b)
            data[y][x][0] = a
            # data[y][x][1] = g
            # data[y][x][2] = b
            #data[y][x] = a
    return data


#padding 0 surround the picture data
def padding(convdata, times):
    #print(convdata)
    pdata = [[[0 for i in range(len(convdata[0][0]))]for x in range(len(convdata[0])+(2*times))]for y in range(len(convdata)+(2*times))]
    for i in range(len(convdata[0][0])):
        for y in range(len(convdata)):
            for x in range(len(convdata[0])):
                pdata[y + times][x + times][i] = convdata[y][x][i]
    return pdata

def ranKernal():
    ker = [[0 for x in range(3)]for y in range(3)]
    for i in range(3):
        for j in range(3):
            ker[i][j] = random.randint(0, 1)
    #print('-----kernel')
    #print(ker)
    return ker

def firstConv(data, times, strides):

    #psteps = ( n(s-1) - s + f ) / 2
    padding_steps = int((len(data)*(strides - 1) - strides + 3) / 2)
    #padding
    pdata = padding(data, padding_steps)
    #calculate the output data size : ((n + 2p - f) / s ) + 1
    sizeConvdataOut = int(len(data) + 2*padding_steps - 3 / strides ) + 1
    #print(sizeConvdataOut)
    #convdataOut[y][x][i = conv times(16)]
    convdata = [[[0 for rgb in range(3)] for x in range(sizeConvdataOut)] for y in range(sizeConvdataOut)]
    convdataOut = [[[0 for i in range(times)]for x in range(sizeConvdataOut)]for y in range(sizeConvdataOut)]
    kernalArr = []

    for i in range(times):
        kernal = ranKernal()
        kernalArr.append(kernal)
        for y in range(1, len(pdata) - 1):
            for x in range(1, len(pdata[y]) - 1):
                convdata[y - 1][x - 1][0] = pdata[y - 1][x - 1][0] * kernal[0][0] + pdata[y - 1][x][0] * kernal[0][
                    1] + pdata[y - 1][x + 1][0] * kernal[0][2] + \
                                            pdata[y][x - 1][0] * kernal[1][0] + pdata[y][x][0] * kernal[1][1] + \
                                            pdata[y][x + 1][0] * kernal[1][2] + \
                                            pdata[y + 1][x - 1][0] * kernal[2][0] + pdata[y + 1][x][0] * kernal[2][
                                                1] + pdata[y + 1][x + 1][0] * kernal[2][2]
                convdata[y - 1][x - 1][1] = pdata[y - 1][x - 1][1] * kernal[0][0] + pdata[y - 1][x][1] * kernal[0][
                    1] + \
                                            pdata[y - 1][x + 1][1] * kernal[0][2] + \
                                            pdata[y][x - 1][1] * kernal[1][0] + pdata[y][x][1] * kernal[1][1] + \
                                            pdata[y][
                                                x + 1][1] * kernal[1][2] + \
                                            pdata[y + 1][x - 1][1] * kernal[2][0] + pdata[y + 1][x][1] * kernal[2][
                                                1] + \
                                            pdata[y + 1][x + 1][1] * kernal[2][2]
                convdata[y - 1][x - 1][2] = pdata[y - 1][x - 1][2] * kernal[0][0] + pdata[y - 1][x][2] * kernal[0][
                    1] + \
                                            pdata[y - 1][x + 1][2] * kernal[0][2] + \
                                            pdata[y][x - 1][2] * kernal[1][0] + pdata[y][x][2] * kernal[1][1] + \
                                            pdata[y][
                                                x + 1][2] * kernal[1][2] + \
                                            pdata[y + 1][x - 1][2] * kernal[2][0] + pdata[y + 1][x][2] * kernal[2][
                                                1] + \
                                            pdata[y + 1][x + 1][2] * kernal[2][2]
                #merge RGB data : convdata[][][] to convdataOut[][][]
                convdataOut[y - 1][x - 1][i] = convdata[y - 1][x - 1][0] + convdata[y - 1][x - 1][1] + \
                                               convdata[y - 1][x - 1][2]
                x += strides - 1
            y += strides - 1
                #test trans to gray
                # gray = (convdata[y - 1][x - 1][0] + convdata[y - 1][x - 1][1] + convdata[y - 1][x - 1][2]) / 3
                # convdataGray[y - 1][x - 1][0] , convdataGray[y - 1][x - 1][1] , convdataGray[y - 1][x - 1][2] = gray, \
                #                                                                                             gray, gray

    #print(convdataOut)
    return convdataOut

def nextConv(data, times, strides):
    # psteps = ( n(s-1) - s + f ) / 2
    padding_steps = int(ceil((len(data) * (strides - 1) - strides + 3) / 2))
    #print(len(data))
    #print(padding_steps)
    # padding
    pdata = padding(data, padding_steps)
    # calculate the output data size : ((n + 2p - f) / s ) + 1
    sizeConvdataOut = int((len(data) + (2 * padding_steps) - 3 )/ strides) + 1
    #pdata = condataout (layer1)
    #pdata formate: [y][x][i = conv times(16)]
    convdataL2 = [[0 for x in range(len(pdata[1]))] for y in range(len(pdata))]
    convdataOut = [[[0 for i in range(times)] for x in range(sizeConvdataOut)] for y in range(sizeConvdataOut)]
    kernalArr = []
    #print(len(convdataL2))

    #compress pdata(y*x*16dim) to condataL2(y*x*1dim)
    for y in range(len(pdata)):
        for x in range(len(pdata[y])):
            convdataL2[y][x] = sum(pdata[y][x])


    #convolution
    for i in range(times):
        kernal = ranKernal()
        kernalArr.append(kernal)
        for y in range(1, len(convdataL2) - strides , strides):
            for x in range(1, len(convdataL2[y]) - strides , strides):
                #print(y,x)
                convdataOut[int((y - 1)/strides)][int((x - 1)/strides)][i] = convdataL2[y - 1][x - 1] * kernal[0][0] + \
                                               convdataL2[y - 1][x] * kernal[0][1] + \
                                               convdataL2[y - 1][x + 1] * kernal[0][2] + \
                                               convdataL2[y][x - 1] * kernal[1][0] + \
                                               convdataL2[y][x] * kernal[1][1] + \
                                               convdataL2[y][x + 1] * kernal[1][2] + \
                                               convdataL2[y + 1][x - 1] * kernal[2][0] + \
                                               convdataL2[y + 1][x] * kernal[2][1] + \
                                               convdataL2[y + 1][x + 1] * kernal[2][2]
    #print(convdataOut)
    return convdataOut

def lastmerge(convdata):
    #[y][x][] to [y][x]
    convedData = [[0 for x in range(len(convdata[0]))] for y in range(len(convdata))]
    for y in range(len(convdata)):
        for x in range(len(convdata[y])):
            convedData[y][x] = sum(convdata[y][x])
            #print(sum(convdata[y][x]))
    #print(convedData)
    return convedData


def pooling(conved):
    #print(len(conved),len(conved[0]))
    #print(conved)
    # conved format: [y][x]
    poolingOut = [[0 for x in range(len(conved[0])-2)] for y in range(len(conved)-2)]

    for y in range(1, len(conved) - 1):
        for x in range(1, len(conved[y]) - 1):
            poolingOut[y-1][x-1] = max(conved[y - 1][x - 1], conved[y - 1][x], conved[y - 1 ][x + 1],
                                       conved[y][x - 1], conved[y][x], conved[y][x + 1],
                                       conved[y + 1][x - 1], conved[y + 1][x], conved[y + 1][x + 1])
    return poolingOut

def flatten2DTo1D(poolingOut):
    flattenData = []
    for y in range(len(poolingOut)):
        for x in range(len(poolingOut[0])):
            flattenData.append(poolingOut[y][x])
    return flattenData

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
    lossSum *= (1/2)
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
    for case in range(10,60):
        for num in range(3,6):
            data = importMNISTToData('../pic/{}/{}.png'.format(num,case)) # square picture only
            convdata = nextConv(data,3 ,2)
            nextconvdata = nextConv(convdata, 3, 2)
            output = lastmerge(nextconvdata)
            pooling1 = pooling(output)
            flatten = flatten2DTo1D(pooling1)
            img_arr.append(flatten)
            if (num == 3) :
                ans_arr.append([1,0,0])
            elif(num == 4):
                ans_arr.append([0,1,0])
            else:
                ans_arr.append([0,0,1])
                
    input_size = 676
    hidden_size = 400
    output_size = 3
    learning_rate = 0.005

    # 隨機初始化權重
    #np.random.seed(0)
    weights_input_hidden = np.random.uniform(-0.1,0.1,size=(input_size, hidden_size))
    weights_hidden_output = np.random.uniform(-0.1,0.1,size=(hidden_size, output_size))
    
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
        if(loss <= 0.001):
            break
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

#=======================================================================================以下測試
    img_arr2 = []
    for case in range(140,190):
        for num in range(3,6):
            data = importMNISTToData('../pic/{}/{}.png'.format(num,case)) # square picture only
            convdata = nextConv(data,3 ,2)
            nextconvdata = nextConv(convdata, 3, 2)
            output = lastmerge(nextconvdata)
            pooling1 = pooling(output)
            flatten = flatten2DTo1D(pooling1)
            img_arr2.append(flatten)
    predicted_output = sigmoid(np.dot(sigmoid(np.dot(img_arr2, weights_input_hidden)), weights_hidden_output))
    
    ans = [0,0,0]
    count = 0
    for i in range(0,len(predicted_output)):
        if (i%3 == 0):
            ans = [1,0,0]
        elif(i%3 == 1):
            ans = [0,1,0]
        else:
            ans = [0,0,1]
        mm = max(predicted_output[i][0],predicted_output[i][1],predicted_output[i][2])
        arr = []
        for k in range(0,3):
            if (predicted_output[i][k] == mm):
                arr.append(1)
            else : 
                arr.append(0)
        if (arr == ans):
            count+=1
            
    print(str(count) + '/' + str(len(predicted_output)))
    print(str(count/len(predicted_output)*100)+'%')