
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Image data matrix format : data[y:height][x:width][3:rgb]

from PIL import Image
import numpy as np
import random

width = 0
height = 0

def importImgToData(path):
    global height, width
    orgImg = Image.open(path)
    pxMatrix = np.asarray(orgImg)
    print("pic dim :",pxMatrix.ndim)
    #print(pxMatrix[200][200])

    height = orgImg.height
    width = orgImg.width
    data = [[[0 for rgb in range(3)] for x in range(width)] for y in range(height)]
    #print(pxMatrix[282][648])

    for y in range(height):
        for x in range(width):
            r, g, b = pxMatrix[y][x]
            #print(r, g, b)
            data[y][x][0] = r
            data[y][x][1] = g
            data[y][x][2] = b
    #print(data[282][648])

    return data

def padding(data):
    pdata = [[[0 for rgb in range(3)] for x in range(width+2)] for y in range(height+2)]
    for y in range(height):
        for x in range(width):
            for rgb in range(3):
                pdata[y+1][x+1][rgb] = data[y][x][rgb]
    return pdata


def firstConv(pdata):
    #format [straight][horizontal]
    convdata = [[[0 for rgb in range(3)] for x in range(width)] for y in range(height)]
    XKernal = [[1, 0, 1],
               [0, 1, 0],
               [1, 0, 1]]
    for y in range(1, len(pdata)-1):
        for x in range(1, len(pdata[y])-1):
            convdata[y-1][x-1][0] = pdata[y-1][x-1][0]*XKernal[0][0] + pdata[y-1][x][0]*XKernal[0][1] + pdata[y-1][x+1][0]*XKernal[0][2] +\
                                    pdata[y][x - 1][0] * XKernal[1][0] + pdata[y][x][0]*XKernal[1][1] + pdata[y][x+1][0]*XKernal[1][2] +\
                                    pdata[y+1][x - 1][0] * XKernal[2][0] + pdata[y+1][x][0]*XKernal[2][1] + pdata[y+1][x+1][0]*XKernal[2][2]
            convdata[y - 1][x - 1][1] = pdata[y - 1][x - 1][1] * XKernal[0][0] + pdata[y - 1][x][1] * XKernal[0][1] + \
                                        pdata[y - 1][x + 1][1] * XKernal[0][2] + \
                                        pdata[y][x - 1][1] * XKernal[1][0] + pdata[y][x][1] * XKernal[1][1] + pdata[y][
                                            x + 1][1] * XKernal[1][2] + \
                                        pdata[y + 1][x - 1][1] * XKernal[2][0] + pdata[y + 1][x][1] * XKernal[2][1] + \
                                        pdata[y + 1][x + 1][1] * XKernal[2][2]
            convdata[y - 1][x - 1][2] = pdata[y - 1][x - 1][2] * XKernal[0][0] + pdata[y - 1][x][2] * XKernal[0][1] + \
                                        pdata[y - 1][x + 1][2] * XKernal[0][2] + \
                                        pdata[y][x - 1][2] * XKernal[1][0] + pdata[y][x][2] * XKernal[1][1] + pdata[y][
                                            x + 1][2] * XKernal[1][2] + \
                                        pdata[y + 1][x - 1][2] * XKernal[2][0] + pdata[y + 1][x][2] * XKernal[2][1] + \
                                        pdata[y + 1][x + 1][2] * XKernal[2][2]
    return convdata


def ranKernal():
    ker = [[0 for x in range(3)]for y in range(3)]
    for i in range(3):
        for j in range(3):
            ker[i][j] = random.randint(0, 1)
    print(ker)
    return ker

def conv(pdata, times):
    #convdataOut[i = conv times(16)][y][x][rgb]
    convdata = [[[0 for rgb in range(3)] for x in range(width)] for y in range(height)]
    convdataGray = [[[0 for rgb in range(3)] for x in range(width)] for y in range(height)]
    #convdataOut = [[[0 for x in range(width)]for y in range(height)]for i in range(times)]
    for i in range(times):
        kernal = ranKernal()
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
                #merge RGB data
                # convdataOut[i][y - 1][x - 1] = convdata[y - 1][x - 1][0] + convdata[y - 1][x - 1][1] + \
                #                                convdata[y - 1][x - 1][2]

                #test trans to gray
                gray = (convdata[y - 1][x - 1][0] + convdata[y - 1][x - 1][1] + convdata[y - 1][x - 1][2]) / 3
                convdataGray[y - 1][x - 1][0] , convdataGray[y - 1][x - 1][1] , convdataGray[y - 1][x - 1][2] = gray, \
                                                                                                             gray, gray

    return convdataGray



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = importImgToData('../pic/train.jpg')
    pdata = padding(data)
    #firstconvdata = firstConv(pdata)
    convdata = conv(pdata, 1)
    print(convdata[0][0])
    #print(convdata)
    testimg = Image.fromarray(np.uint8(convdata))
    testimg.show()






