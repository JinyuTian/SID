import argparse
import time
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn as nn
from utils import *
import datetime
import sys
from dataloaders import cifar10

from get_model import get_model,get_dual_model

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-TB', default=128, type=int,
                    metavar='N', help='Training batch size')
parser.add_argument('-VB', default=50, type=int,
                    metavar='N', help='Testing batch size')

parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-2, type=float,
                   help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--augment', default=True, type=bool)
parser.add_argument('--dataset',
                    help='Target Database',
                    default='Imagenet', type=str)
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='', type=str)
parser.add_argument('--domain',
                    help='pixel or wavelets domain',
                    default='pixel', type=str)
parser.add_argument('--wave',
                    help='pixel or wavelets domain',
                    default='sym17', type=str)
parser.add_argument('--optimizer',
                    help='pixel or wavelets domain',
                    default='db3', type=str)
parser.add_argument('--gpuid', default='0', type=str)
parser.add_argument('--net_type', type=str, help='resnet|cifar10')

parser.set_defaults(net_type='resnet')
parser.set_defaults(dataset='cifar10')
parser.set_defaults(TB=80)
parser.set_defaults(VB=20)
parser.set_defaults(lr=1e-6)
parser.set_defaults(weight_decay=0.005)
parser.set_defaults(domain='FD')
parser.set_defaults(epochs=300)
parser.set_defaults(augment=True)
parser.set_defaults(resume=False)
parser.set_defaults(optimizer='SGD')
parser.set_defaults(start_epoch=0)
parser.set_defaults(FixedWave='weighted')
parser.set_defaults(resumepath='')
best_prec1 = 0


def main():
    global args, best_prec1,VISDOM
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    print('Training Model: {}'.format(args.net_type))
    print('Opt:{opt}; Learning Rate: {lr:.1e}; Weight Decay: {wd:.1e}'.format(opt=args.optimizer,lr=args.lr,wd=args.weight_decay))
    #### Experiment Record
    time = datetime.datetime.now()
    ExperimentID = datetime.datetime.strftime(time, '%m-%d')
    ScriptName = os.path.basename(sys.argv[0].split('/')[-1].split('.')[0])
    SaveDir = 'ExperimentRecord/'+ExperimentID+'/'+ScriptName +'-'+ args.net_type + '-' + str(args.lr)+'_'+str(args.weight_decay)
    CreateFolder(SaveDir)
    Log = SaveDir+'/Log.txt'
    ExpLog = open(Log, 'w')
    ExpLog.close()
    ExpLog = open(Log, 'a')
    args.save_dir = SaveDir
    ExpLog.write('LR:' + str(args.lr) + ';' + ' WD:' + str(args.weight_decay) + '; ' + \
                 'TrainBatch:' + str(args.TB) + '\n')


    model = GET_MODEL(args)
    model.cuda()
    train_loader, val_loader, num_classes = get_data(args)

    ####### define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    #### Initialization
    if 0:
        for p in model.parameters():
            p.requires_grad = True
            try:
                torch.nn.init.normal_(p, a=0, mode='fan_in')
            except:
                torch.nn.init.normal_(p, mean=0.0, std=1.0)


    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    ####### Training
    OldNamePath = args.save_dir
    Titer = 0; Viter = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        Titer = train(train_loader, model, criterion, optimizer, epoch,ExpLog,Titer)
        # evaluate on validation set
        prec1,Viter = validate(val_loader, model, criterion,ExpLog,Viter)
        # remember best prec@1 and save checkpoint
        if prec1 > best_prec1:
            torch.save({
                'args': args,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': prec1,
            }, os.path.join(OldNamePath, 'checkpoint_{}.tar'.format(prec1)))
            best_prec1 = prec1

def train(train_loader, model, criterion, optimizer, epoch,F,Iter):
    """
        Run one train epoch
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    i = 0
    for input, target in train_loader:
        # measure data loading time
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda().long()
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # for p in model.parameters():
        #     print(p.grad)
        #     break
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        # measure elapsed time

        if i % 1 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1))

            F.write('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader),loss=losses, top1=top1)+'\n')
            F.flush()
        i = i+1
    return Iter

def validate(val_loader, model, criterion,F,Iter):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # for i, (input, target) in enumerate(val_loader):
    i = 0
    for input, target in val_loader:
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda().long()

        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), loss=losses,
                      top1=top1))
            F.write('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader),  loss=losses,
                      top1=top1)+'\n')
            F.flush()
        i = i + 1
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, Iter

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def GET_MODEL(args):
    model = get_model(args,args.net_type)
    if os.path.exists(args.resumepath) and args.resume:
        checkpoint = torch.load(args.resumepath)
        if type(checkpoint) == dict:
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                DICT = checkpoint['state_dict'].copy()
                for key in checkpoint['state_dict'].keys():
                    if key.split('.')[0] == 'module':
                        newkey = key.split('module.')[1]
                        DICT.update({newkey: DICT.pop(key)})
                    elif key.split('.')[1] == 'module':
                        newkey = key.split('module.')[0]+key.split('module.')[1]
                        DICT.update({newkey: DICT.pop(key)})
                model.load_state_dict(DICT)
        else:
            model.load_state_dict(checkpoint)
    return model

def GET_PMODEL(args,net_type):
    args.resume = './pre_trained/' + args.domain+net_type + '_' + args.dataset + '.pth'
    model = get_model(args,net_type)
    checkpoint = torch.load(args.resume)
    if type(checkpoint) == dict:
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            DICT = checkpoint['state_dict'].copy()
            for key in checkpoint['state_dict'].keys():
                if key.split('.')[0] == 'module':
                    newkey = key.split('module.')[1]
                    DICT.update({newkey: DICT.pop(key)})
                elif key.split('.')[1] == 'module':
                    newkey = key.split('module.')[0]+key.split('module.')[1]
                    DICT.update({newkey: DICT.pop(key)})
            model.load_state_dict(DICT)
    else:
        model.load_state_dict(checkpoint)
    return model

def get_data(args):
    if args.dataset == 'cifar10':
        train_loader, val_loader, num_classes = cifar10(augment=args.augment, batch_size=args.TB)
    elif args.dataset == 'SVHN':
        train_loader, val_loader, num_classes = SVHN(augment=args.augment, batch_size=args.TB)
    return train_loader, val_loader, num_classes
if __name__ == '__main__':
    main()
