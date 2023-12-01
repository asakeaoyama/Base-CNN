# Process of this program:
# read image -> conv (decrease method and max pooling) -> merge and flatten -> output(convedData list and kernel sets)


from PIL import Image
import numpy as np
import random
from math import ceil


class FixedKernelConventionalConv:
    width = 0
    height = 0
    data = 0
    # kernel Set [convlayer][kernels per layer][kernelY][kernelX]
    kerSet = None

    finalOutput = 0

    def __init__(self, path,kernelSet, isMNIST, layers, times, strides):
        self.kerSet = kernelSet
        if (isMNIST == True):
            self.data = self.importMNISTToData(path)  # square picture only
        else:
            self.data = self.importImgToData(path)  # square picture only
        # convdata = firstConv(data, times , strides)
        convdata = self.data
        for i in range(layers):
            convdata = self.conventionalConv(convdata, times, strides, i)
            convdata = self.poolingInConving(convdata)
            #print("---------------------------------------------")
        conved = self.lastmerge(convdata)
        #pooling = self.pooling(conved)
        flatten = self.flatten2DTo1D(conved)
        self.finalOutput = flatten

    def importMNISTToData(self, path):
        orgImg = Image.open(path)
        pxMatrix = np.asarray(orgImg)
        # print("pic dim :", pxMatrix.ndim)

        self.height = orgImg.height
        self.width = orgImg.width
        data = [[[0 for rgb in range(3)] for x in range(self.width)] for y in range(self.height)]
        # print(height, width)

        for y in range(self.height):
            for x in range(self.width):
                r, g, b, a = pxMatrix[y][x]
                # print(r, g, b)
                data[y][x][0] = a
                # data[y][x][1] = g
                # data[y][x][2] = b
                # data[y][x] = a

        # print(data)

        self.data = data
        return data

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

    def padding(self, convdata):
        # print(convdata)
        pdata = [[[0 for i in range(len(convdata[0][0]))] for x in range(len(convdata[0]) + 1)] for y in
                 range(len(convdata) + 1)]
        for i in range(len(convdata[0][0])):
            for y in range(len(convdata)):
                for x in range(len(convdata[0])):
                    pdata[y][x][i] = convdata[y][x][i]
        return pdata

    def ranKernel(self):
        ker = [[0 for x in range(3)] for y in range(3)]
        for i in range(3):
            for j in range(3):
                ker[i][j] = random.randint(0, 1)
        #print(ker)
        return ker

    def conventionalConv(self, data, times, strides, convLayers):
        #print("datalen:", len(data))
        #print("dataDepth:", len(data[0][0]))
        # padding
        # padding trigger : len(data)-1 % stride == 1 then padding 1 step on left and below
        pdata = None
        if (len(data) - 1) % strides == 1:
            pdata = self.padding(data)
            #print("padding: True")
        else:
            pdata = data
            #print("padding: False")
        # calculate the conv output data size : ((n - 2) / s) + 1
        sizeConvdataOut = int(((len(data) - 2) / strides) + 1)
        #print("predictOutputLan:", sizeConvdataOut)
        # pdata format: [y][x][i = conv times(16)]
        # convdataL2 format: [y][x]
        convdataL2 = [[0 for x in range(len(pdata[1]))] for y in range(len(pdata))]
        convdataOut = [[[0 for i in range(times)] for x in range(sizeConvdataOut)] for y in range(sizeConvdataOut)]

        #print("preConvedOutputLen:", len(convdataL2))

        # compress pdata(y*x*16dim) to condataL2(y*x*1dim)
        for y in range(len(pdata)):
            for x in range(len(pdata[0])):
                convdataL2[y][x] = sum(pdata[y][x])

        # convolution
        for i in range(times):
            kernel = self.kerSet[convLayers][i]
            #print(kernel)
            for y in range(1, len(convdataL2) - strides, strides):
                for x in range(1, len(convdataL2[y]) - strides, strides):
                    convdataOut[int((y - 1) / strides)][int((x - 1) / strides)][i] = convdataL2[y - 1][x - 1] * \
                                                                                     kernel[0][0] + \
                                                                                     convdataL2[y - 1][x] * kernel[0][
                                                                                         1] + \
                                                                                     convdataL2[y - 1][x + 1] * \
                                                                                     kernel[0][2] + \
                                                                                     convdataL2[y][x - 1] * kernel[1][
                                                                                         0] + \
                                                                                     convdataL2[y][x] * kernel[1][1] + \
                                                                                     convdataL2[y][x + 1] * kernel[1][
                                                                                         2] + \
                                                                                     convdataL2[y + 1][x - 1] * \
                                                                                     kernel[2][0] + \
                                                                                     convdataL2[y + 1][x] * kernel[2][
                                                                                         1] + \
                                                                                     convdataL2[y + 1][x + 1] * \
                                                                                     kernel[2][2]
        # print(convdataOut)
        #print("postConvedDataLen:", len(convdataOut))
        return convdataOut

    def lastmerge(self, convdata):
        # [y][x][i] to [y][x]
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
        poolingOut = [[0 for x in range(len(conved[0]) - 2)] for y in range(len(conved) - 2)]

        for y in range(1, len(conved) - 1):
            for x in range(1, len(conved[y]) - 1):
                poolingOut[y - 1][x - 1] = max(conved[y - 1][x - 1], conved[y - 1][x], conved[y - 1][x + 1],
                                               conved[y][x - 1], conved[y][x], conved[y][x + 1],
                                               conved[y + 1][x - 1], conved[y + 1][x], conved[y + 1][x + 1])
        return poolingOut

    def poolingInConving(self, data):
        # data format: [y][x][i] to output [y-2][x-2][i]
        output = [[[0 for i in range(len(data[0][0]))]for x in range(len(data[0]) - 2)]for y in range(len(data)-2)]
        for y in range(1, len(data) - 1):
            for x in range(1, len(data[0]) - 1):
                for i in range(len(data[0][0])):
                    output[y - 1][x - 1][i] = max(data[y - 1][x - 1][i], data[y - 1][x][i], data[y - 1][x + 1][i],
                                               data[y][x - 1][i], data[y][x][i], data[y][x + 1][i],
                                               data[y + 1][x - 1][i], data[y + 1][x][i], data[y + 1][x + 1][i])
        #print(output)
        return output

    def flatten2DTo1D(self, poolingOut):
        flattenData = []
        for y in range(len(poolingOut)):
            for x in range(len(poolingOut[0])):
                flattenData.append(poolingOut[y][x])
        return flattenData

    def getKernelSet(self):
        return self.kerSet


if __name__ == '__main__':
    kernel = [1, 0, 1],[0, 1, 0],[1, 0, 1]
    kernelSet = [[kernel for i in range(8)]for j in range(2)]
    # format: conv(path, isMNIST, layers, times per layer, strides)
    #convly = ConventionalConv('../pic/train.jpg', False, 3, 16, 2)
    convly = ConventionalConv('../pic/4/0.png', kernelSet, True, 2, 8, 2)
    print(convly.finalOutput)
    print("length of the output: ", len(convly.finalOutput))
