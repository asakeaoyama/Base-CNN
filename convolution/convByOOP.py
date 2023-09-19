from PIL import Image
import numpy as np
import random
from math import ceil

class conv():
    width = 0
    height = 0
    data = 0

    finalOutput = 0

    def __init__(self, path, layers, times, strides):

        self.data = self.importImgToData(path)  # square picture only
        # convdata = firstConv(data, times , strides)
        convdata = self.nextConv(self.data, times, strides)
        nextconvdata = self.nextConv(convdata, times, strides)
        conved = self.lastmerge(nextconvdata)

        pooling = self.pooling(conved)
        self.finalOutput = pooling

    def importImgToData(self, path):
        orgImg = Image.open(path)
        pxMatrix = np.asarray(orgImg)
        # print("pic dim :", pxMatrix.ndim)

        self.height = orgImg.height
        self.width = orgImg.width
        data = [[[0 for rgb in range(3)] for x in range(self.width)] for y in range(self.height)]
        # print(height, width)

        for y in range(self.height):
            for x in range(self.width):
                r, g, b = pxMatrix[y][x]
                # print(r, g, b)
                data[y][x][0] = r
                data[y][x][1] = g
                data[y][x][2] = b

        self.data = data
        return data

    def padding(self, convdata, times):
        # print(convdata)
        pdata = [[[0 for i in range(len(convdata[0][0]))] for x in range(len(convdata[0]) + (2 * times))] for y in
                 range(len(convdata) + (2 * times))]
        for i in range(len(convdata[0][0])):
            for y in range(len(convdata)):
                for x in range(len(convdata[0])):
                    pdata[y + times][x + times][i] = convdata[y][x][i]
        return pdata

    def ranKernal(self):
        ker = [[0 for x in range(3)] for y in range(3)]
        for i in range(3):
            for j in range(3):
                ker[i][j] = random.randint(0, 1)
        print(ker)
        return ker

    def firstConv(self, data, times, strides):

        # psteps = ( n(s-1) - s + f ) / 2
        padding_steps = int((len(data) * (strides - 1) - strides + 3) / 2)
        # padding
        pdata = self.padding(data, padding_steps)
        # calculate the output data size : ((n + 2p - f) / s ) + 1
        sizeConvdataOut = int(len(data) + 2 * padding_steps - 3 / strides) + 1
        # print(sizeConvdataOut)
        # convdataOut[y][x][i = conv times(16)]
        convdata = [[[0 for rgb in range(3)] for x in range(sizeConvdataOut)] for y in range(sizeConvdataOut)]
        convdataGray = [[[0 for rgb in range(3)] for x in range(self.width)] for y in range(self.height)]
        convdataOut = [[[0 for i in range(times)] for x in range(sizeConvdataOut)] for y in range(sizeConvdataOut)]
        kernalArr = []

        for i in range(times):
            kernal = self.ranKernal()
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
                    # merge RGB data : convdata[][][] to convdataOut[][][]
                    convdataOut[y - 1][x - 1][i] = convdata[y - 1][x - 1][0] + convdata[y - 1][x - 1][1] + \
                                                   convdata[y - 1][x - 1][2]
                    x += strides - 1
                y += strides - 1
                # test trans to gray
                # gray = (convdata[y - 1][x - 1][0] + convdata[y - 1][x - 1][1] + convdata[y - 1][x - 1][2]) / 3
                # convdataGray[y - 1][x - 1][0] , convdataGray[y - 1][x - 1][1] , convdataGray[y - 1][x - 1][2] = gray, \
                #                                                                                             gray, gray

        # print(convdataOut)
        return convdataOut

    def nextConv(self, data, times, strides):
        # psteps = ( n(s-1) - s + f ) / 2
        padding_steps = int(ceil((len(data) * (strides - 1) - strides + 3) / 2))
        print(len(data))
        print(padding_steps)
        # padding
        pdata = self.padding(data, padding_steps)
        # calculate the output data size : ((n + 2p - f) / s ) + 1
        sizeConvdataOut = int((len(data) + (2 * padding_steps) - 3) / strides) + 1
        print(sizeConvdataOut)
        # pdata = condataout (layer1)
        # pdata formate: [y][x][i = conv times(16)]
        convdataL2 = [[0 for x in range(len(pdata[1]))] for y in range(len(pdata))]
        convdataOut = [[[0 for i in range(times)] for x in range(sizeConvdataOut)] for y in range(sizeConvdataOut)]
        kernalArr = []
        print(len(convdataL2))

        # compress pdata(y*x*16dim) to condataL2(y*x*1dim)
        for y in range(len(pdata)):
            for x in range(len(pdata[y])):
                convdataL2[y][x] = sum(pdata[y][x])

        # convolution
        for i in range(times):
            kernal = self.ranKernal()
            kernalArr.append(kernal)
            for y in range(1, len(convdataL2) - strides, strides):
                for x in range(1, len(convdataL2[y]) - strides, strides):
                    convdataOut[int((y - 1) / strides)][int((x - 1) / strides)][i] = convdataL2[y - 1][x - 1] * \
                                                                                     kernal[0][0] + \
                                                                                     convdataL2[y - 1][x] * kernal[0][
                                                                                         1] + \
                                                                                     convdataL2[y - 1][x + 1] * \
                                                                                     kernal[0][2] + \
                                                                                     convdataL2[y][x - 1] * kernal[1][
                                                                                         0] + \
                                                                                     convdataL2[y][x] * kernal[1][1] + \
                                                                                     convdataL2[y][x + 1] * kernal[1][
                                                                                         2] + \
                                                                                     convdataL2[y + 1][x - 1] * \
                                                                                     kernal[2][0] + \
                                                                                     convdataL2[y + 1][x] * kernal[2][
                                                                                         1] + \
                                                                                     convdataL2[y + 1][x + 1] * \
                                                                                     kernal[2][2]
        # print(convdataOut)
        return convdataOut

    def lastmerge(self, convdata):
        # [y][x][] to [y][x]
        convedData = [[0 for x in range(len(convdata[0]))] for y in range(len(convdata))]
        for y in range(len(convdata)):
            for x in range(len(convdata[y])):
                convedData[y][x] = sum(convdata[y][x])
                # print(sum(convdata[y][x]))
        # print(convedData)
        return convedData

    def pooling(self, conved):
        # print(len(conved),len(conved[0]))
        # print(conved)
        # conved format: [y][x]
        poolingOut = [[0 for x in range(self.width - 2)] for y in range(self.height - 2)]

        for y in range(1, len(conved) - 1):
            for x in range(1, len(conved[y]) - 1):
                poolingOut[y - 1][x - 1] = max(conved[y - 1][x - 1], conved[y - 1][x], conved[y - 1][x + 1],
                                               conved[y][x - 1], conved[y][x], conved[y][x + 1],
                                               conved[y + 1][x - 1], conved[y + 1][x], conved[y + 1][x + 1])
        return poolingOut


if __name__ == '__main__':
    convly = conv('../pic/10x10.PNG', 2, 3, 2)
    print(convly.finalOutput)





    #print(len(pooling), len(pooling[1]))
    #print(pooling)