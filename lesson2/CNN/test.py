# -*- coding: utf-8 -*-

"""
@author: Min Li
@e-mail: limin93@ihep.ac.cn
@file: test.py
@time: 2023/9/4 10:56
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
in_channels, out_channels = 5, 10
width, height =100, 100
kernel_size = 3
batch_size = 1

input = torch.randn(batch_size, in_channels, width, height)
conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

output = conv_layer(input)
print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)

input = [  3, 4, 6,5, 7,
                2, 4, 6, 8 ,2,
                1, 6, 7, 8, 4,
                9, 7, 4, 6, 2,
                3, 7, 5, 4, 1]
input = torch.Tensor(input).view(1,1,5,5)
print(input)
conv_layer = torch.nn.Conv2d(1,1,kernel_size=3,  stride=1, bias=False)
kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3)
conv_layer.weight.data = kernel.data
output = conv_layer(input)
print(output)

####Max pooing Layer
input = [3, 4, 6, 5,
                2, 4, 6, 8,
                1, 6, 7, 8,
                9, 7, 4, 6,
            ]

input = torch.Tensor(input).view(1,1,4,4)
maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)
output = maxpooling_layer(input)
print(output)

