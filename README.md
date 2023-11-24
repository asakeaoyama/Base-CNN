# Basic Convolutional Neural Network Development

INTRODUCTION:
Hi, This is our project implemented as a basic convolutional neural network. The feature is just identify a word on a picture being imported. The stucture of the project is separate to two parts: Convolutional Layer and Neural Network Layer. But they are intergrated in merely a CNN_NKNU.py file in the root path. You can check it out and alter the picture path if needed.

ENVIRONMENT:
Recommend IDE: pycharm 
Environment: Anaconda venv.
Needed package include "numpy", "pillow", "random", that's all.

Format of Using Classes:
  Convolution Layer:
    Same padding:
      = conv(PICTURE PATH, isMNIST, Layers, Times per Layer, Strides)
    Conventional Convolution:
      = FixedKernelConventionalConv(PATH, KernelSet, isMNIST, Layers, Times per Layer, Strides)

Since 2023,05,16
At this moment, there is only convolution feature separate with neural network.
