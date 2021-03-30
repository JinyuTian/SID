import argparse
import torch.nn.parallel
import pandas as pd
import torch.optim
import torch.utils.data
import torch.nn as nn
from get_model import get_model,get_dual_model
from utils import *
import sys
import calculate_log as callog
import glob

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('-TB', default=128, type=int)
parser.add_argument('-TB2', default=128, type=int)
parser.add_argument('-VB', default=50, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float)
parser.add_argument('--momentum', default=0.2, type=float)
parser.add_argument('--TSR', default=0.8, type=float)
parser.add_argument('--weight-decay', '--wd', default=0, type=float)
parser.add_argument('--print-freq', default=1, type=int)
parser.add_argument('--PDresume', default='', type=str)
parser.add_argument('--FDresume', default='', type=str)
parser.add_argument('--Dualresume', default='', type=str)
parser.add_argument('--show', default='', type=bool)
parser.add_argument('-CT', type=bool,help='Continue to train a model')
parser.add_argument('--Val', type=bool, help='Valiate the generalization ability')
parser.add_argument('--SID_data_type', type=str, help='attack target dataset')
parser.add_argument('--save-dir', type=str, help='save checkpoint')
parser.add_argument('--ExDate', type=str, help='experiment date')
parser.add_argument('--optimizer', type=str,help='SGD or Adam')
parser.add_argument('--SID_adv_type', type=str, help='attack methods: FGSM|DeepFool|CWL2|BIM')
parser.add_argument('--domain', type=str, default='PD', help='Frequence domian or pixel domian')
parser.add_argument('--Model', type=str, help='adversarial samples detection model')
parser.add_argument('--gpuid', type=str)
parser.add_argument('--ScriptName', type=str)
parser.add_argument('--shuffle', default=False, type=bool)
parser.add_argument('--retrain', default=False, type=bool)
parser.add_argument('--Verify', type=bool, help='whether or not to verify the trained model')
parser.add_argument('--SID_net_type', type=str, help='Target Model: resnet|VggBn')
parser.add_argument('--ATP2', default=0.9, type=float, help='parameter of attack methods')
parser.add_argument('--wavemode', type=str, help='the mode to combination four channel of wavelets (average or append)')
parser.add_argument('--wave', type=str, help='wavelets used to train a FDmodel')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--FeatureLayers', default=[], type=list,
                    help='layers used to calculate LID features and MHB features (Comparision methods)')
parser.add_argument('--SID_magnitude', default=0., type=float,
                    help='perturbation magnitude')

parser.add_argument('--num_class', default=10, type=int)

parser.set_defaults(CT=False)
parser.set_defaults(ATP2 = 0.1)
parser.set_defaults(ExDate='10-23')
parser.set_defaults(weight_decay=5e-3)
parser.set_defaults(show=False)
parser.set_defaults(TB=100)
parser.set_defaults(retrain=False)
parser.set_defaults(TB2=300)
parser.set_defaults(TSR=0.6)
parser.set_defaults(lr=0.005)
parser.set_defaults(Val=True)
parser.set_defaults(Verify = False)
parser.set_defaults(shuffle = True)
parser.set_defaults(FDmode='append')
parser.set_defaults(ATP1 = 0.005)
parser.set_defaults(Model = 'Dual4')
parser.set_defaults(optimizer = 'SGD')
parser.set_defaults(gpuid = '1')
parser.set_defaults(momentum = 0.8)
parser.set_defaults(Succ = True)
parser.set_defaults(wave = 'sym17')
parser.set_defaults(FixedWave = 'Fixed1')
parser.set_defaults(AdvNoise=0.05)
parser.set_defaults(Calibration=0)
best_prec1 = 0

def main():
    global args, DIR,TESTDIRDIR
    args = parser.parse_args()
    args.ScriptName = os.path.basename(sys.argv[0].split('/')[-1].split('.')[0])
    args.AE_source = 'ExperimentRecord/KnownAttack/'
    args.save_dir = os.path.join('ExperimentRecord/',args.ScriptName)
    os.makedirs(args.save_dir,exist_ok=True)

    #### create DataFrames to save results
    DataFrames = {}
    for path in os.listdir(args.AE_source):
        net_type = path.split('_')[0]
        data_type = path.split('_')[1]
        adv_type = path.split('_')[2]
        magnitude = path.split('_')[3]
        if not net_type+'_'+data_type in DataFrames.keys():
            DataFrames[net_type + '_' + data_type] = {}
            DataFrames[net_type + '_' + data_type]['sources'] = []
        DataFrames[net_type+'_'+data_type]['sources'].append(adv_type+'_'+magnitude)

    for key in DataFrames.keys():
        sources = sorted(DataFrames[key]['sources'])
        DataFrames[key]['data'] = pd.DataFrame(np.zeros([len(sources), len(sources)]), index=sources, columns=sources)

    #### create SID
    SID = get_dual_model(C_Number=3, num_class=10)
    SID.cuda()

    for SID_name in os.listdir(args.AE_source):
        args.SID_net_type = SID_name.split('_')[0]
        args.SID_data_type = SID_name.split('_')[1]
        args.SID_adv_type = SID_name.split('_')[2]
        args.SID_magnitude = SID_name.split('_')[3]
        args.SID_AE_source = args.SID_adv_type+'_'+args.SID_magnitude
        Detector_path = os.path.join(args.AE_source,SID_name)
        DetectorResume = os.path.join(Detector_path, 'checkpoint.tar')
        if not os.path.exists(DetectorResume):
            continue
        else:
            Dualcheckpoint = torch.load(DetectorResume)
            SID.load_state_dict(Dualcheckpoint['state_dict'])

        for adv_source in os.listdir(args.AE_source):
            source_net_type = adv_source.split('_')[0]
            source_data_type = adv_source.split('_')[1]
            if source_net_type == args.SID_net_type and source_data_type == args.SID_data_type:
                print("Using SID ({}) to detect AEs of source: {}".format(SID_name,adv_source))
                DataFrames = Detect(adv_source,SID,args,DataFrames)

    for key in DataFrames.keys():
        DataFrames[key]['data'].to_csv('{}.csv'.format(key))
        print('Save generalizability results of SIDs on: {}'.format(key))

def Detect(adv_source,SID,args,DataFrames):
    LogitsPath = os.path.join(os.path.join(args.AE_source, adv_source),'Logits')
    BLogits, BLabel, CTarget, CLogits, TrainIndex, ValIndex = GetStackLogitValues(LogitsPath)
    results = ValClassifer(BLogits, BLabel, SID, ValIndex, args.save_dir,PRINT=False)
    target_AE_source = adv_source.split('_')[2]+'_'+adv_source.split('_')[3]
    DataFrames[args.SID_net_type + '_' + args.SID_data_type]['data'][args.SID_AE_source][target_AE_source] = '{:.2f}'.format(100*results['TMP']['AUROC'])

    for mtype in ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100. * results['TMP']['TNR']), end='')
    print(' {val:6.2f}'.format(val=100. * results['TMP']['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100. * results['TMP']['DTACC']), end='')
    print(' {val:6.2f}'.format(val=100. * results['TMP']['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100. * results['TMP']['AUOUT']), end='')

    return DataFrames

def TrainClassifer(BLogits,BLabel,model,CLogits,CTarget,criterion, optimizer, epoch,TrainIndex):
    """
        Data: includes clean, adv, noisy
        CrtSuccImgs: 100% predicted images by PDmodel
        CrtSuccLabs: label of CrtSuccImgs
        Calibration: let it go
    """
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    TotalDataScale = len(TrainIndex)
    TrainIndex = TrainIndex[np.random.permutation(TotalDataScale)]
    # for i, (input, target) in enumerate(train_loader):
    i,j= 0,0
    while not len(TrainIndex) == 0:
        index = np.random.permutation(CLogits.shape[1])[0:args.TB2]
        CPDX = CLogits[0,index]
        CFDX = CLogits[1,index]
        gnt = CTarget[index]
        if len(TrainIndex) < args.TB2:
            FDx = torch.from_numpy(BLogits[1,TrainIndex[0:]].astype(dtype=np.float32))
            PDx = torch.from_numpy(BLogits[0,TrainIndex[0:]].astype(dtype=np.float32))
            target = torch.from_numpy(BLabel[TrainIndex[0:]])
        else:
            PDx = torch.from_numpy(BLogits[0,TrainIndex[0:args.TB2]].astype(dtype=np.float32))
            FDx = torch.from_numpy(BLogits[1,TrainIndex[0:args.TB2]].astype(dtype=np.float32))
            target = torch.from_numpy(BLabel[TrainIndex[0:args.TB2]])
        target = target.cuda()
        target = target.cuda()
        PDx = torch.autograd.Variable(PDx).cuda()
        FDx = torch.autograd.Variable(FDx).cuda()
        target_var = torch.autograd.Variable(target).long()

        ####
        PDprec1 = np.nonzero((np.argmax(CPDX,1) - gnt) == 0)[0].shape[0]/CPDX.shape[0]
        FDprec1 = np.nonzero((np.argmax(CFDX,1) - gnt) == 0)[0].shape[0]/CFDX.shape[0]
        output = model(PDx, FDx)
        loss = criterion(output, target_var)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.data, PDx.shape[0])
        top1.update(prec1[0], PDx.shape[0])
        # measure elapsed time
        if i % args.print_freq == 0:
            if (i + 1) * args.TB2 > TotalDataScale:
                r = '\rEpoch: [{0}][{1}/{2}] Loss {loss.avg:.4f} ' \
                    ' Prec@1 {top1.avg:.3f} PDAC {PDAC: .2f} FDAC {FDAC: .2f}'.format(
                    epoch, TotalDataScale, TotalDataScale,
                    loss=losses, top1=top1, PDAC=PDprec1,FDAC=FDprec1)
            else:
                r = '\rEpoch: [{0}][{1}/{2}] Loss {loss.avg:.4f}' \
                    ' Prec@1 {top1.avg:.3f} PDAC {PDAC: .2f} FDAC {FDAC: .2f}'.format(
                    epoch, (i + 1) * args.TB2, TotalDataScale,
                    loss=losses, top1=top1, PDAC=PDprec1,FDAC=FDprec1)
            sys.stdout.write(r)
        i = i+1
        TrainIndex = TrainIndex[args.TB2:]
    print()

def GetLogitValues(Data, label, PDmodel, FDmodel, args,Natural=True,Selected=False):
    """
        This function split the clean Data, Noisy data, and Advserarial data into three classes:
        PF_PosImgs = 0: clean images and noisy images which are correctly predicted by both PDmodel and FDmodel
        P_PosImgs = 1: clean images which are correctly predicted by both PDmodel but incorrectly predicted by FDmodel
        NegImgs = 2: Adversarial images

        Variables:
        FDmodel: Frequcency domain model
        PDmodel: Pixel domain model
        PFCrtImgs: Clean images correctly predicted by both PDmodel and FDmodel
        PCrtImgs: Clean images correctly predicted by the PDmodel only
        PAdvImgs: Adv images successfully attacked the PDmodel only
        PFAdvImgs: Adv images successfully attacked both the PDmodel and the FDmodel
        PFNsyImgs: Noisy images correctly predicted by both PDmodel and FDmodel
        PNsyImgs: Noisy images correctly predicted by both PDmodel only
    """
    if isinstance(Data, torch.Tensor):
        Data = Data.numpy()
    if isinstance(label,torch.Tensor):
        label = label.numpy()
    FDmodel.eval()
    PDmodel.eval()
    TrainIndex = np.arange(0,Data.shape[0])
    Scale = len(TrainIndex)
    PFLogits =np.zeros([2,2,args.num_class])
    PLogits = np.zeros([2,2,args.num_class])
    PF_Targets = np.zeros([2])
    P_Targets = np.zeros([2])
    Flag = np.zeros([2])
    #### Loop according to batch size args.TB
    i = 0
    while not len(TrainIndex) == 0:
        if len(TrainIndex) < args.TB:
            TempIndex = TrainIndex[0:]
            inputs = torch.from_numpy(Data[TempIndex])
            target = torch.from_numpy(label[TempIndex])
        else:
            TempIndex = TrainIndex[0:args.TB]
            inputs = torch.from_numpy(Data[TempIndex])
            target = torch.from_numpy(label[TempIndex])
        input_var = torch.autograd.Variable(inputs).cuda().float()
        target_var = torch.autograd.Variable(target.cuda()).long()
        PDx = PDmodel(input_var)
        FDx = FDmodel(input_var)
        PDprec1, Ppred = accuracy(PDx.data, target_var)
        FDprec1, Fpred = accuracy(FDx.data, target_var)
        PDx = PDx.detach().cpu().numpy()
        FDx = FDx.detach().cpu().numpy()
        if Natural == True:
            PFCrtImgsIdx = np.nonzero((Ppred - Fpred).cpu().numpy() == 0)[1]
            PCrtImgsIdx = np.nonzero((Ppred - Fpred).cpu().numpy())[1]
            if Selected == True:
                SuccClfFlag = np.nonzero(Ppred.cpu().numpy() - target.cpu().numpy() == 0)[1]
                Flag = np.concatenate((Flag,TempIndex[SuccClfFlag]),axis=0)
                PFCrtImgsIdx = np.intersect1d(PFCrtImgsIdx,SuccClfFlag)
                PCrtImgsIdx = np.intersect1d(PCrtImgsIdx,SuccClfFlag)
            TPFLogits = np.zeros([2,len(PFCrtImgsIdx),PDx.shape[-1]])
            TPLogits = np.zeros([2,len(PCrtImgsIdx),PDx.shape[-1]])
            TPFLogits[0,:] = PDx[PFCrtImgsIdx]
            TPFLogits[1,:] = FDx[PFCrtImgsIdx]
            TPLogits[0,:] = PDx[PCrtImgsIdx]
            TPLogits[1,:] = FDx[PCrtImgsIdx]
            TPF_Targets = target.numpy()[PFCrtImgsIdx]
            TP_Targets = target.numpy()[PCrtImgsIdx]
            PFLogits = np.concatenate((PFLogits,TPFLogits),axis=1)
            PLogits = np.concatenate((PLogits,TPLogits),axis=1)
            PF_Targets = np.concatenate((PF_Targets,TPF_Targets))
            P_Targets = np.concatenate((P_Targets,TP_Targets))
        else:
            TPFLogits = np.zeros([2,PDx.shape[0],PDx.shape[-1]])
            TPFLogits[0,:] = PDx
            TPFLogits[1,:] = FDx
            TPF_Targets = target.numpy()
            if Selected == True:
                SuccAdvFlag = np.nonzero(Ppred.cpu().numpy() - target.cpu().numpy() != 0)[1]
                Flag = np.concatenate((Flag,TempIndex[SuccAdvFlag]),axis=0)
                TPFLogits = TPFLogits[:,SuccAdvFlag,:]
                TPF_Targets = TPF_Targets[SuccAdvFlag]
            PFLogits = np.concatenate((PFLogits,TPFLogits),axis=1)
            PF_Targets = np.concatenate((PF_Targets, TPF_Targets))
        if i % args.print_freq == 0:
            if (i + 1) * args.TB > Scale:
                r = '\rTest: [{0}/{1}]'.format(Scale, Scale )
            else:
                r = '\rTest: [{0}/{1}]'.format((i + 1) * args.TB, Scale )
            sys.stdout.write(r)
        i = i+1
        TrainIndex = TrainIndex[args.TB:]
    if Natural == True:
        PFLogits = PFLogits[:,2:, ]
        PLogits = PLogits[:,2:,]
        PF_Targets = PF_Targets[2:]
        P_Targets = P_Targets[2:]
        Flag = Flag[2:]
        Label = np.concatenate((0 * np.ones(PFLogits.shape[1]), 1 * np.ones(PLogits.shape[1])))
        return PFLogits, PLogits, Label, PF_Targets, P_Targets,Flag
    else:
        PFLogits = PFLogits[:,2:, ]
        PF_Targets = PF_Targets[2:]
        Label = 2 * np.ones(PFLogits.shape[1])
        Flag = Flag[2:]
        return PFLogits, Label, PF_Targets,Flag
    # Label = torch.cat((0*torch.ones(PF_PosImgs.shape[0]),1*torch.ones(P_PosImgs.shape[0]),
    #                    2*torch.ones(PF_NegImgs.shape[0]),3*torch.ones(P_NegImgs.shape[0])),0)
    # return PF_PosImgs, P_PosImgs, PF_NegImgs, P_NegImgs, Label

def ValClassifer2(BLogits,BLabel,model,criterion,TrainIndex):
    """
        Data: includes clean, adv, noisy
        CrtSuccImgs: 100% predicted images by PDmodel
        CrtSuccLabs: label of CrtSuccImgs
        Calibration: let it go
    """
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    TotalDataScale = len(TrainIndex)
    TrainIndex = TrainIndex[np.random.permutation(TotalDataScale)]
    # for i, (input, target) in enumerate(train_loader):
    i,j= 0,0
    while not len(TrainIndex) == 0:
        if len(TrainIndex) < args.TB2:
            FDx = torch.from_numpy(BLogits[1,TrainIndex[0:]].astype(dtype=np.float32))
            PDx = torch.from_numpy(BLogits[0,TrainIndex[0:]].astype(dtype=np.float32))
            target = torch.from_numpy(BLabel[TrainIndex[0:]])
        else:
            PDx = torch.from_numpy(BLogits[0,TrainIndex[0:args.TB2]].astype(dtype=np.float32))
            FDx = torch.from_numpy(BLogits[1,TrainIndex[0:args.TB2]].astype(dtype=np.float32))
            target = torch.from_numpy(BLabel[TrainIndex[0:args.TB2]])
        target = target.cuda()
        target = target.cuda()
        PDx = torch.autograd.Variable(PDx).cuda()
        FDx = torch.autograd.Variable(FDx).cuda()
        target_var = torch.autograd.Variable(target).long()

        ####
        output = model(PDx, FDx)
        loss = criterion(output, target_var)
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.data, PDx.shape[0])
        top1.update(prec1[0], PDx.shape[0])
        # measure elapsed time
        if i % args.print_freq == 0:
            if (i + 1) * args.TB2 > TotalDataScale:
                r = '\rEpoch: [{0}][{1}/{2}] Loss {loss.avg:.4f} ' \
                    ' Prec@1 {top1.avg:.3f}'.format(
                    'Test', TotalDataScale, TotalDataScale,
                    loss=losses, top1=top1)
            else:
                r = '\rEpoch: [{0}][{1}/{2}] Loss {loss.avg:.4f}' \
                    ' Prec@1 {top1.avg:.3f}'.format(
                    'Test', (i + 1) * args.TB2, TotalDataScale,
                    loss=losses, top1=top1)
            sys.stdout.write(r)
        i = i+1
        TrainIndex = TrainIndex[args.TB2:]
    return top1.avg

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_data(args):
    clean_data = np.load(
        '{0}/{1}_{2}_{5}/clean_data_{1}_{2}_{3}_{4:.2f}.npy'.format(args.outf, args.net_type, args.dataset, args.adv_type,
                                                                args.AdvNoise,args.adv_type))
    adv_data = np.load(
        '{0}/{1}_{2}_{5}/adv_data_{1}_{2}_{3}_{4:.2f}.npy'.format(args.outf, args.net_type, args.dataset, args.adv_type,
                                                              args.AdvNoise, args.adv_type))
    noisy_data = np.load(
        '{0}/{1}_{2}_{5}/noisy_data_{1}_{2}_{3}_{4:.2f}.npy'.format(args.outf, args.net_type, args.dataset, args.adv_type,
                                                                args.AdvNoise,args.adv_type))
    label = np.load(
        '{0}/{1}_{2}_{5}/label_{1}_{2}_{3}_{4:.2f}.npy'.format(args.outf, args.net_type, args.dataset, args.adv_type,
                                                           args.AdvNoise,args.adv_type))



    return clean_data,adv_data, noisy_data, label

def get_auroc(output,target_var,SaveDir):
    Bioutput = torch.zeros([output.shape[0], 2])
    Bioutput[:, 0] = torch.max(output[:, 0:2], 1)[0]
    Bioutput[:, 1] = output[:, 2]
    target_var[np.nonzero(target_var.cpu().numpy() == 1)] = 0
    target_var[np.nonzero(target_var.cpu().numpy() == 2)] = 1
    Bioutput = torch.nn.Softmax(dim=1)(Bioutput)
    y_pred = Bioutput.detach().cpu().numpy().astype(np.float64)[:, 1]
    Y = target_var.detach().cpu().numpy().astype(np.float64)
    num_samples = Y.shape[0]
    l1 = open('%s/confidence_TMP_In.txt' % SaveDir, 'a')
    l2 = open('%s/confidence_TMP_Out.txt' % SaveDir, 'a')
    for i in range(num_samples):
        if Y[i] == 0:
            l1.write("{}\n".format(-y_pred[i]))
        else:
            l2.write("{}\n".format(-y_pred[i]))
    l1.flush()
    l2.flush()
    results = callog.metric(SaveDir, ['TMP'])
    return results

def processBar(epoch ,args, TotalDataScale,losses,top1,i):
    if (i + 1) * args.TB > TotalDataScale:
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

def detection_performance(y_pred, Y,outf):
    """
    Measure the detection performance
    return: detection metrics
    """
    y_pred = y_pred.detach().cpu().numpy().astype(np.float64)[:, 1]
    Y = Y.detach().cpu().numpy().astype(np.float64)
    num_samples = Y.shape[0]
    l1 = open('%s/confidence_TMP_In.txt'%outf, 'w')
    l2 = open('%s/confidence_TMP_Out.txt'%outf, 'w')

    for i in range(num_samples):
        if Y[i] == 0:
            l1.write("{}\n".format(-y_pred[i]))
        else:
            l2.write("{}\n".format(-y_pred[i]))
    l1.close()
    l2.close()
    results = callog.metric(outf, ['TMP'])
    return results

def GET_MODEL(args,net_type):
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

def get_adv_noisy(net_type,dataset):
    if net_type == 'VggBn' and dataset == 'cifar10':
        adv_test_list = ['BIM','DeepFool','FGSM','CWL2']
        adv_noisy_list = [0.97,0.51,.28,.73]
    if net_type == 'VggBn' and dataset == 'svhn':
        adv_test_list = ['BIM','DeepFool','FGSM','CWL2']
        adv_noisy_list = [1.05,.32,.33,0.67]
    if net_type == 'resnet' and dataset == 'cifar10':
        adv_test_list = ['BIM','DeepFool','CWL2']
        adv_noisy_list = [0.49,0.28,0.46]
    if net_type == 'resnet' and dataset == 'svhn':
        adv_test_list = ['BIM','DeepFool','FGSM','CWL2']
        adv_noisy_list = [1.03,.76,.33,0.81]
    return adv_test_list, adv_noisy_list

def GetStackLogitValues(LogitsPath):

    Size = np.load(LogitsPath + '/Size.npy')
    BLogits = np.load(LogitsPath+'/BLogits.npy')
    BLabel = np.load(LogitsPath+'/BLabel.npy')
    CTarget = np.load(LogitsPath+'/CTarget.npy')
    CLogits = np.load(LogitsPath+'/CLogits.npy')

    TS0 = int(np.floor(args.TSR * Size[0]))
    TS1 = int(np.floor(args.TSR * Size[1]))
    TS2 = int(np.floor(args.TSR * Size[2]))
    CPFindex = np.random.permutation(Size[0])
    CPindex = np.random.permutation(Size[1])
    Advindex = np.random.permutation(Size[2])
    TrainIndex = np.concatenate((CPFindex[0:TS0],CPindex[0:TS1]+Size[0],CPFindex[0:TS0]+Size[0]+Size[1],
                                 CPindex[0:TS1]+2*Size[0]+Size[1],
                                 Advindex[0:TS2]+2*Size[1]+2*Size[0]),axis=0)
    TestIndex = np.concatenate((CPFindex[TS0:],CPindex[TS1:]+Size[0],CPFindex[TS0:]+Size[0]+Size[1],
                                CPindex[TS1:]+2*Size[0]+Size[1],
                                Advindex[TS2:]+2*Size[1]+2*Size[0]),axis=0)

    return BLogits,BLabel,CTarget,CLogits,TrainIndex,TestIndex

def ValClassifer(BLogits,BLabel, model,Index,SaveDir,PRINT=False):
    Index = Index[np.random.permutation(Index.shape[0])]
    open('%s/confidence_TMP_In.txt' % SaveDir, 'w').close()
    open('%s/confidence_TMP_Out.txt' % SaveDir, 'w').close()
    model.eval()
    TotalDataScale = len(Index)
    i = 0
    while not len(Index) == 0:
        if len(Index) < args.TB:
            FDx = torch.from_numpy(BLogits[1, Index[0:]].astype(dtype=np.float32))
            PDx = torch.from_numpy(BLogits[0, Index[0:]].astype(dtype=np.float32))
            target = torch.from_numpy(BLabel[Index[0:]])
        else:
            PDx = torch.from_numpy(BLogits[0, Index[0:args.TB]].astype(dtype=np.float32))
            FDx = torch.from_numpy(BLogits[1, Index[0:args.TB]].astype(dtype=np.float32))
            target = torch.from_numpy(BLabel[Index[0:args.TB]])
        target = target.cuda()
        target = target.cuda()
        PDx = torch.autograd.Variable(PDx).cuda()
        FDx = torch.autograd.Variable(FDx).cuda()
        target_var = torch.autograd.Variable(target).long()
        output = model(PDx, FDx)
        output = output.float()
        Index = Index[args.TB:]
        results = get_auroc(output, target_var, SaveDir)
        auroc = results['TMP']['AUROC']

        if PRINT:
            if (i + 1) * args.TB > TotalDataScale:
                r = '\rEpoch: [Test][{0}/{1}] ' \
                    ' auroc{auroc: .3f}'.format(TotalDataScale, TotalDataScale, auroc=auroc*100)
            else:
                r = '\rEpoch: [Test][{0}/{1}] ' \
                    ' auroc{auroc: .3f}'.format((i + 1) * args.TB, TotalDataScale, auroc=auroc*100)
            sys.stdout.write(r)

        i = i+1

    return results

if __name__ == '__main__':
    main()
