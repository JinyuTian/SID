import os
import torch
import numpy as np
for file in os.listdir('/home/jinyu/Temp/adv_output/resnet_cifar10_BIM/0.41_0.005'):
    path2 = '/home/jinyu/Temp/adv_output/resnet_cifar10_BIM/0.41_0.005/'+file
    A = np.load(path2)
    np.save(path2,A)