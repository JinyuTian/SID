import os
import shutil
import torchvision.transforms as transforms
import numpy as np
import sys
def RecordImages(Images,Labels,ImagePathes,ImFP,ImLP,Ind,SAVE = False):
    ## ImFP: The path for saving Images.
    ## ImLP: The log for recording the path and lable of each image.
    ## Ind: The index of images to be saved in Images
    Log = open(ImLP, 'a')
    for i in Ind:
        Image = InverseTransform(Images[i].detach().cpu().squeeze())
        ImageName = ImagePathes[i].split('/')[-1]
        if SAVE:
            ImagePath = os.path.join(ImFP, ImageName)
            Image.save(ImagePath,quality=95)
            Log.write(ImagePath + ',' + str(Labels[i].cpu().detach().numpy().astype(int)) + '\n')
        else:
            Log.write(ImagePathes[i] + ',' + str(Labels[i].cpu().detach().numpy().astype(int)) + '\n')

    Log.close()
    return ImFP,ImLP

def CreateFolder(ImageFolder):
    if not os.path.exists(ImageFolder):
        os.makedirs(ImageFolder)
    if os.path.exists(ImageFolder):
        shutil.rmtree(ImageFolder)
        os.makedirs(ImageFolder)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res,pred

def InverseTransform(x):
    y = transforms.Compose([
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.ToPILImage()
    ])(x)
    return y

def SplitTVData(ImageLog,TSRatio,VSRatio,SaveDir):
    F = open(ImageLog)
    lines = F.readlines()
    TrainScale = np.floor(TSRatio*len(lines))
    TestScale = np.floor(VSRatio * len(lines))
    TrainLogPath = os.path.join(SaveDir,'TrainLog.txt')
    TestLogPath = os.path.join(SaveDir, 'TestLog.txt')
    open(TrainLogPath,'w').close(); open(TestLogPath, 'w').close()
    TrainLog = open(TrainLogPath, 'a');
    TestLog = open(TestLogPath, 'a')
    for i,line in enumerate(lines):
        if i < TrainScale:
            TrainLog.write(line)
        elif i > TrainScale and i < TestScale+TrainScale:
            TestLog.write(line)
    return TrainLogPath,TestLogPath

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    ce = - np.mean(np.log(predictions) * targets,axis=1)
    return ce

def asymmetricKL(P,Q):
    return sum(P * np.log(P / Q)) #calculate the kl divergence between P and Q

def symmetricalKL(P,Q):
    return (asymmetricKL(P,Q)+asymmetricKL(Q,P))/2.00


def processBar(epoch ,args, TotalDataScale,losses,top1,i):
    if i == 0:
        r = '\nEpoch: [{0}][{1}/{2}] Loss {loss.avg:.4f}' \
            ' Prec@1 {top1.avg:.3f}'.format(
            epoch, (i + 1) * args.TB, TotalDataScale,
            loss=losses, top1=top1)
    elif (i + 2) * args.TB > TotalDataScale:
        r = '\rEpoch: [{0}][{1}/{2}] Loss {loss.avg:.4f} ' \
            ' Prec@1 {top1.avg:.3f}'.format(
            epoch, TotalDataScale, TotalDataScale,
            loss=losses, top1=top1)
    else:
        r = '\rEpoch: [{0}][{1}/{2}] Loss {loss.avg:.4f}' \
            ' Prec@1 {top1.avg:.3f}'.format(
            epoch, (i + 1) * args.TB, TotalDataScale,
            loss=losses, top1=top1)
    sys.stdout.write(r)