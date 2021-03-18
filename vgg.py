'''
Modified from https://github.com/pytorch/vision.git
'''
import torch
import math
import numpy as np
import torch.nn as nn
import pywt.data
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19','FD_vgg19_bn'
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features,NClasses,shape=512 * 7 * 7,Feature_Lists=[1,2,3,4,5]):
        super(VGG, self).__init__()
        self.Feature_Lists = Feature_Lists
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(shape, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, NClasses),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def feature_list(self, x):
        layers = self.Feature_Lists
        if layers[-1] == 'F':
            layers = layers[0:-1]
        y = self.forward(x)
        out_list = []
        count = -1
        for module in self.features._modules.values():
            x = module(x)
            if isinstance(module,nn.Conv2d):
                count += 1
                if count in layers:
                    out_list.append(x)
            if len(out_list) == len(layers):
                break
        if self.Feature_Lists[-1] == 'F':
            out_list.append(y)
        return y, out_list

    def intermediate_forward(self, x, layer_index):
        layers = self.Feature_Lists
        if layers[layer_index]=='F':
            x = self.forward(x)
            return x
        COUNT = layers[layer_index]
        counter = -1
        for module in self.features._modules.values():
            x = module(x)
            if isinstance(module,nn.Conv2d):
                counter += 1
            if counter == COUNT:
                break
        return x

class FDVGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features,NClasses,shape=512 * 7 * 7,Feature_Lists=[]):
        super(FDVGG, self).__init__()
        self.Feature_Lists = Feature_Lists
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(shape, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, NClasses),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.wavelets(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


    def wavelets(self,x):
        output = torch.zeros(x.shape[0],x.shape[1]*4,int(x.shape[2]/2),int(x.shape[3]/2))
        output = output.cuda()
        xfm = DWTForward(J=1, wave='haar', mode='symmetric').cuda()
        Yl, Yh = xfm(x)
        output[:,[0, 4, 8],:] = Yl
        output[:, 1:4, :] = Yh[0][:, 0, :]
        output[:, 5:8, :] = Yh[0][:, 1, :]
        output[:, 9:12, :] = Yh[0][:, 2, :]
        return output

def make_layers(cfg, batch_norm=False,chanel=3):
    layers = []
    in_channels = chanel
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn(NClasses,chanel=3,shape=512*3*3,Feature_Lists=[]):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True,chanel=chanel),NClasses=NClasses,
               shape=shape,Feature_Lists=Feature_Lists)

def FD_vgg19_bn(NClasses,chanel=3,shape=512*3*3,args=[]):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return FDVGG(make_layers(cfg['E'], batch_norm=True,chanel=chanel),NClasses=NClasses,
               shape=shape)