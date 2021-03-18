import torchvision.datasets as datasets
import pywt.data
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import os
import numpy as np



def default_loader(path):
    return Image.open(path).convert('RGB')

class DatasetForDetection(Dataset):
    def __init__(self,ImageInfo,Transform,mode='pixel',DataScale=0):
        ImagePathList = []
        ImageLabelList = []
        GTLabelList = []
        F = open(ImageInfo)
        self.FError = open('WrongImage.txt', 'w')
        self.FError.close()
        self.mode = mode
        self.FError = open('WrongImage.txt', 'a')
        lines = F.readlines()
        for line in lines:
            ImagePath = line.split(',')[0]
            ImageLabel = line.split(',')[-1].split('\n')[0]
            GTLabel = line.split(',')[1].split(':')[1]
            ImagePathList.append(ImagePath)
            ImageLabelList.append(ImageLabel)
            GTLabelList.append(GTLabel)
        self.ImagePathList = ImagePathList
        self.ImageLabelList = ImageLabelList
        self.GTLabelList = GTLabelList
        if not DataScale == 0:
            self.ImagePathList = ImagePathList[:DataScale]
            self.ImageLabelList = ImageLabelList[:DataScale]
        # self.loader = loader
        self.Transform = Transform
    def loader(self,ImagePath):
        if self.mode == 'pixel':
            img = Image.open(ImagePath).convert('RGB')
            img = self.Transform(img)
            return img
        elif self.mode == 'wavelets':
            img = Image.open(ImagePath).convert('RGB')
            img = self.Transform(img)
            img = np.array(img)
            IMG = np.zeros([256,256,img.shape[2]*4])
            for i in range(img.shape[0]):
                data = img[i,:,:].squeeze()
                coeffs2 = pywt.dwt2(data,'haar',mode='symmetric')
                if i == 0:
                    LL = coeffs2[0].reshape([1,coeffs2[0].shape[0],coeffs2[0].shape[1]])
                    LH = coeffs2[1][0].reshape([1, coeffs2[0].shape[0], coeffs2[0].shape[1]])
                    HL = coeffs2[1][1].reshape([1, coeffs2[0].shape[0], coeffs2[0].shape[1]])
                    HH = coeffs2[1][2].reshape([1, coeffs2[0].shape[0], coeffs2[0].shape[1]])
                    IMG = np.concatenate((LL,LH,HL,HH),axis=0)
                else:
                    LL = coeffs2[0].reshape([1,coeffs2[0].shape[0],coeffs2[0].shape[1]])
                    LH = coeffs2[1][0].reshape([1, coeffs2[0].shape[0], coeffs2[0].shape[1]])
                    HL = coeffs2[1][1].reshape([1, coeffs2[0].shape[0], coeffs2[0].shape[1]])
                    HH = coeffs2[1][2].reshape([1, coeffs2[0].shape[0], coeffs2[0].shape[1]])
                    IMG = np.concatenate((IMG,LL,LH,HL,HH),axis=0)
            return IMG

    def __getitem__(self, index):
        ImagePath = self.ImagePathList[index]
        # PATH = '/home/pubuser/TJY/TIANCHI/TrainData/00058/af1894bce72d6981f6854afa60178981.jpg'
        label = float(self.ImageLabelList[index])
        label = torch.tensor(int(label), dtype=torch.long)
        GT = float(self.GTLabelList[index])
        GT = torch.tensor(int(GT), dtype=torch.long)
        try:
            img = self.loader(ImagePath)
        except:

            self.FError.write(ImagePath)
        return img, label, ImagePath,GT

    def __len__(self):
        return len(self.ImagePathList)

class MyDataset(Dataset):
    def __init__(self,ImageInfo,Transform,loader,DataScale=0):
        ImagePathList = []
        ImageLabelList = []
        F = open(ImageInfo)
        self.FError = open('WrongImage.txt', 'w')
        self.FError.close()
        self.FError = open('WrongImage.txt', 'a')
        lines = F.readlines()
        for line in lines:
            ImagePath = line.split(',')[0]
            ImageLabel = line.split(',')[1].split('\n')[0]
            ImagePathList.append(ImagePath)
            ImageLabelList.append(ImageLabel)
        self.ImagePathList = ImagePathList
        self.ImageLabelList = ImageLabelList
        if not DataScale == 0:
            self.ImagePathList = ImagePathList[:DataScale]
            self.ImageLabelList = ImageLabelList[:DataScale]
        self.loader = loader
        self.Transform = Transform
    def __getitem__(self, index):
        ImagePath = self.ImagePathList[index]
        # PATH = '/home/pubuser/TJY/TIANCHI/TrainData/00058/af1894bce72d6981f6854afa60178981.jpg'
        label = float(self.ImageLabelList[index])
        label = torch.tensor(int(label), dtype=torch.long)
        try:
            img = self.loader(ImagePath)
            img = self.Transform(img)
        except:
            self.FError.write(ImagePath)
        return img, label, # ImagePath

    def __len__(self):
        return len(self.ImagePathList)

def ModelChoice(ModelName,TEST):
    if ModelName == 'Inc':
        rootpath = os.path.join('./', 'InceptionV3')
        Model = models.inception_v3(pretrained=TEST)
        imputsize = [299, 299]
    elif ModelName == 'VGG19_bn':
        rootpath = os.path.join('./', 'VGG19_bn')
        Model = models.vgg19_bn(pretrained=TEST)
        imputsize = [244, 244]

    elif ModelName == 'VGG19':
        rootpath = os.path.join('./', 'VGG19')
        Model = models.vgg19(pretrained=TEST)
        imputsize = [244, 244]

    elif ModelName == 'VGG16':
        rootpath = os.path.join('./', 'VGG19')
        Model = models.vgg16(pretrained=TEST)
        imputsize = [244, 244]

    elif ModelName == 'MyVGG19_bn':
        rootpath = os.path.join('./', 'VGG19_bn')
        Model = models.vgg19_bn(pretrained=TEST)
        imputsize = [244, 244]
    elif ModelName == 'VGG11_bn':
        rootpath = os.path.join('./', 'VGG11_bn')
        Model = models.vgg11_bn(pretrained=TEST)
        imputsize = [244, 244]

    elif ModelName == 'VGG13_bn':
        rootpath = os.path.join('./', 'VGG13_bn')
        Model = models.vgg13_bn(pretrained=TEST)
        imputsize = [244, 244]

    elif ModelName == 'VGG16_bn':
        rootpath = os.path.join('./', 'VGG16_bn')
        Model = models.vgg16_bn(pretrained=TEST)
        imputsize = [244, 244]

    elif ModelName == 'Resnet18':
        rootpath = os.path.join('./', 'Resnet18')
        Model = models.resnet18(pretrained=TEST)
        imputsize = [224, 224]

    elif ModelName == 'Resnet34':
        rootpath = os.path.join('./', 'Resnet34')
        Model = models.resnet34(pretrained=TEST)
        imputsize = [224, 224]

    elif ModelName == 'Resnet50':
        rootpath = os.path.join('./', 'Resnet50')
        Model = models.resnet50(pretrained=TEST)
        imputsize = [224, 224]

    elif ModelName == 'Resnet101':
        rootpath = os.path.join('./', 'Resnet101')
        Model = models.resnet101(pretrained=TEST)
        imputsize = [224, 224]

    elif ModelName == 'Resnet152':
        rootpath = os.path.join('./', 'Resnet152')
        Model = models.resnet152(pretrained=TEST)
        imputsize = [224, 224]

    elif ModelName == 'Densenet121':
        rootpath = os.path.join('./', 'Densenet121')
        Model = models.densenet121(pretrained=TEST)
        imputsize = [244, 244]

    elif ModelName == 'Densenet169':
        rootpath = os.path.join('./', 'Densenet169')
        Model = models.densenet169(pretrained=TEST)
        imputsize = [244, 244]

    elif ModelName == 'Alexnet':
        rootpath = os.path.join('./', 'Alexnet')
        Model = models.alexnet(pretrained=TEST)
        imputsize = [244, 244]


    return Model, rootpath,imputsize

def MydataloaderForDetection(imputsize,ImageInfo,BATCH_SIZE,DataScale,mode,shuffle):
    transform = transforms.Compose([
    transforms.transforms.Resize(imputsize),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

    testData = DatasetForDetection(ImageInfo,transform,mode,DataScale)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=shuffle)
    return testLoader

def Mydataloader(imputsize,ImageInfo,BATCH_SIZE,DataScale,mode,shuffle):
    if imputsize == []:
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    else:
        transform = transforms.Compose([
            transforms.transforms.Resize(imputsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    testData = MyDataset(ImageInfo, transform, default_loader, DataScale)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=shuffle)
    return testLoader

def GeDataloader(args,InputSize,RecordPath):
    if args.dataset == 'CIFAR100':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.dataset == 'Imagenet' or args.dataset == 'imagenet':
        train_info = os.path.join(RecordPath,'TData_{}_{}.txt'.format(args.TRAINSCALE,args.VALSCALE))
        val_info = os.path.join(RecordPath,'TData_{}_{}.txt'.format(args.TRAINSCALE,args.VALSCALE))
        if not os.path.exists(val_info) and not os.path.exists(train_info):
            train_info,val_info = GeDatabaseInfo(args.TRAINSCALE, args.VALSCALE, args.CLASSSCALE,RecordPath)
        train_loader = Mydataloader(InputSize, train_info,args.TB, 0,mode='pixel',shuffle=True,)
        val_loader = Mydataloader(InputSize, val_info, args.VB, 0,mode='pixel',shuffle=True)
    return train_loader,val_loader

def GeWaveletsDataloader(args,InputSize,ExperimentID):


    if args.database == 'CIFAR100':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.database == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.database == 'Imagenet':
        train_info,val_info = GeDatabaseInfo(args.TRAINSCALE, args.VALSCALE, args.CLASSSCALE,ExperimentID)
        train_loader = Mydataloader(InputSize, train_info,args.TB, 0, shuffle=True)
        val_loader = Mydataloader(InputSize, val_info, args.VB, 0, shuffle=True)
    return train_loader,val_loader

def GeDatabaseInfo(TRAINSCALE,VALSCALE,CLASSSCALE,RecordPath):
    ImageInfoFolder = 'ImageInfo'
    if not os.path.exists(ImageInfoFolder):
        os.makedirs(ImageInfoFolder)
    ValDataSource = '/home/jinyu/data/ILSVRC2012_img_val'
    TrainDataSource = '/home/jinyu/data/ILSVRC2012_img_train'
    LabelInfo = {}
    TrainDataSourceFile = os.path.join(RecordPath,'TData_' + str(TRAINSCALE)+'_'+ str(CLASSSCALE)+'.txt')
    ValDataSourceFile = os.path.join(RecordPath,'VData_' + str(VALSCALE)+'_'+ str(CLASSSCALE)+'.txt')
    TrainF = open(TrainDataSourceFile,'w')
    TrainF.close()
    TrainF = open(TrainDataSourceFile,'a')
    ValF = open(ValDataSourceFile,'w')
    ValF.close()
    ValF = open(ValDataSourceFile,'a')
    ClassNameList = os.listdir(ValDataSource)
    ClassNameList = np.sort(ClassNameList)
    for id,ClassName in enumerate(ClassNameList):
        LabelInfo[ClassName] = id
    for ClassName in ClassNameList[:CLASSSCALE]:
        TrainImageNameList = os.listdir(os.path.join(TrainDataSource,ClassName))
        np.random.permutation(TrainImageNameList)
        for ImageName in TrainImageNameList[:TRAINSCALE]:
            ImageLabel = LabelInfo[ClassName]
            ImagePath = os.path.join(TrainDataSource,ClassName,ImageName)
            TrainF.write(ImagePath+','+str(ImageLabel)+'\n')
        ValImageNameList = os.listdir(os.path.join(ValDataSource,ClassName))
        np.random.permutation(ValImageNameList)
        for ImageName in ValImageNameList[:VALSCALE]:
            ImageLabel = LabelInfo[ClassName]
            ImagePath = os.path.join(ValDataSource,ClassName,ImageName)
            ValF.write(ImagePath+','+str(ImageLabel)+'\n')
    return TrainDataSourceFile,ValDataSourceFile