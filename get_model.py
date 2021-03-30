from Models import *
import torch.backends.cudnn as cudnn
import vgg

def get_model(args,net_type):
    try:
        domain = args.domain
    except:
        domain= 'PD'

    if args.dataset == 'cifar10':
        C_Number = 10
        if domain == 'PD':
            in_C = 3
            if net_type == 'VggBn':
                model = PDVgg(in_C,C_Number,batch_norm=True,Feature_Lists=[])
            elif net_type == 'resnet':
                model = ResNet34(num_c=C_Number)
        elif domain == 'FD':
            in_C = 12
            if args.net_type == 'VggBn':
                model = FDVgg(in_C, C_Number, wave='sym17', batch_norm=True)
            elif args.net_type == 'resnet':
                model = FDResNet34(num_c=C_Number,in_C=in_C)
        return model

    elif args.dataset == 'svhn':
        C_Number = 10
        if domain == 'PD':
            in_C = 3
            if net_type == 'VggBn':
                model = PDVgg(in_C, C_Number, batch_norm=True, Feature_Lists=args.FeatureLayers)
            elif net_type == 'resnet':
                model = ResNet34(num_c=C_Number)
        elif domain == 'FD':
            wave = args.wave
            mode = args.FDmode
            in_C = 12
            if net_type == 'VggBn':
                model = FDVgg(in_C, C_Number, wave, mode, batch_norm=True)
            elif net_type == 'resnet':
                model = FDResNet34(num_c=C_Number,in_C=in_C,wave=wave,mode=mode)
        return model

    elif args.dataset == 'imagenet':
        if domain == 'FD':
            model = vgg.__dict__['FD_vgg19_bn'](NClasses=100,chanel=12,shape=512)
            cudnn.benchmark = True
        if domain == 'PD':
            model = vgg.__dict__['vgg19_bn'](NClasses=100,chanel=3,shape=512*3*3,Feature_Lists=args.FeatureLayers)
            cudnn.benchmark = True
        return model


def get_dual_model(C_Number=10, num_class=10):
    model = DualCIFAR4(C_Number=C_Number, num_class=num_class)
    return model




