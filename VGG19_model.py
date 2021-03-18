import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class MyVGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(MyVGG, self).__init__()
        self.features = features
        self.layers = []
        for m in self.features:
            if isinstance(m, nn.Linear) or isinstance(m, L0Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )
        for m in self.classifier:
            if isinstance(m, nn.Linear) or isinstance(m, L0Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def regularization(self):
        ReconstructionError = 0.
        for layer in self.layers:
            if hasattr(layer,'qz_loga'):
                if layer.qz_loga.shape == torch.Size([64]):
                    p = 0.0025
                elif layer.qz_loga.shape == torch.Size([128]):
                    p = 0.3
                elif layer.qz_loga.shape == torch.Size([256]):
                    p = 0.3
                elif layer.qz_loga.shape == torch.Size([512]):
                    p = 0.3
                ReconstructionError += p*layer.ReconstructionError()
        return ReconstructionError.cuda()

    def soft_regularization(self):
        ReconstructionError = 0.
        for layer in self.layers:
            if hasattr(layer,'qz_loga'):
                if layer.qz_loga.shape == torch.Size([64]):
                    p = 0.00001
                elif layer.qz_loga.shape == torch.Size([128]):
                    p = 0.003
                elif layer.qz_loga.shape == torch.Size([256]):
                    p = 0.003
                elif layer.qz_loga.shape == torch.Size([512]):
                    p = 0.003
                ReconstructionError += p*layer.soft_regularization()
        return ReconstructionError.cuda()

def make_layers(in_C,cfg, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_C, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_C = v
    return nn.Sequential(*layers)

def vgg19(pretrained=False,**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = MyVGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model

