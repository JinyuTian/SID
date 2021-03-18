import argparse
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn as nn
from get_model import get_model,get_dual_model
from utils import *
import calculate_log as callog
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('-TB', default=100, type=int, help='Batch size')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float)
parser.add_argument('--momentum', default=0.8, type=float)
parser.add_argument('--TSR', default=0.6, type=float, help='Ratio of traning data to testing data')
parser.add_argument('--weight-decay', '--wd', default=5e-3, type=float)
parser.add_argument('--print-freq', default=1, type=int,help='Frequency of printing log')
parser.add_argument('--PDresume', default='', type=str,help='Checkpoint path of primal classifer')
parser.add_argument('--FDresume', default='', type=str,help='Checkpoint path of dual classifer')
parser.add_argument('--Dualresume', default='', type=str,help='Checkpoint path of SID')
parser.add_argument('--dataset', type=str, help='dataset: CIFAR10|SVHN')
parser.add_argument('--net-type', type=str, help='Target Model: resnet|VggBn')
parser.add_argument('--save-dir', type=str, help='Directory of saving experiment results')
parser.add_argument('--optimizer', type=str,default='SGD',help='SGD | Adam')
parser.add_argument('--adv-type', type=str, help='attack methods: FGSM|DeepFool|CWL2|BIM')
parser.add_argument('--domain', type=str, default='PD', help='Frequence domian or pixel domian')
parser.add_argument('--gpuid', type=str,default='0', help='GPU ids')
parser.add_argument('--ScriptName', type=str,help='Script Name')
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--retrain', default=False, type=bool,help='Retrain detectors')
parser.add_argument('--AdvNoise', default=0., type=float, help='Adversarial perturbation magnitude')
parser.add_argument('--num-class', default=10, help='Number of classes')
parser.add_argument('--adv-source', default=str, help='Source of AEs')


def main():
    global args, DIR,TESTDIRDIR
    args = parser.parse_args()
    adv_sample_source = './adv_output'
    for AdvSource in os.listdir(adv_sample_source):
        args.net_type = AdvSource.split('_')[0]
        args.dataset = AdvSource.split('_')[1]
        args.adv_type = AdvSource.split('_')[2]
        for AdvNoise in os.listdir(os.path.join(adv_sample_source,AdvSource)):
            args.adv_source = os.path.join(os.path.join(adv_sample_source, AdvSource),AdvNoise)
            args.AdvNoise = AdvNoise.split('_')[0]
            TrainDetector(args)

def TrainDetector(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    best_auroc = 0
    args.ScriptName = os.path.basename(sys.argv[0].split('/')[-1].split('.')[0])
    args.save_dir = 'ExperimentRecord/'+args.ScriptName+'/'+ args.net_type + '_' + args.dataset + '_' + \
              args.adv_type + '_' + str(args.AdvNoise)

    #### Create PD and FD models
    args.domain = 'PD'
    PDmodel = GET_MODEL(args,args.net_type)
    args.domain = 'FD'
    FDmodel = GET_MODEL(args,args.net_type)
    PDmodel.cuda()
    FDmodel.cuda()
    FDmodel.eval()
    PDmodel.eval()
    #### Define loss function

    criterion = nn.CrossEntropyLoss().cuda()

    #### get detection model
    DualMOdel = get_dual_model(C_Number=3,num_class=10)
    DualMOdel.cuda()

    for p in DualMOdel.parameters():
        p.requires_grad_()

    #### Define optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(DualMOdel.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(DualMOdel.parameters(), lr=args.lr)

    source = '{}_{}_{}'.format(args.adv_type,args.dataset,args.AdvNoise)
    print('--------------------------------------------------------------------------')
    print('########### Detecting AEs of source: {} ###########'.format(source))

    DetectorResume = os.path.join(args.save_dir, 'checkpoint.tar')
    if os.path.exists(DetectorResume) and not args.retrain:
        Dualcheckpoint = torch.load(DetectorResume)
        DualMOdel.load_state_dict(Dualcheckpoint['state_dict'])
    else:
        clean_data, adv_data, noisy_data, label = get_data(args)
        BLogits,BLabel,CTarget,CLogits,TrainIndex,ValIndex = GetStackLogitValues(clean_data,noisy_data,adv_data,label, PDmodel, FDmodel, args)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40,50,60,70], gamma=1.1, last_epoch=-1)
        print('Train Detector for: {}'.format(source))
        for epoch in range(args.start_epoch, args.epochs):
            scheduler.step()
            TrainClassifer(BLogits,BLabel, DualMOdel,criterion, optimizer, epoch,TrainIndex,args.save_dir)
            print(' ')
            results = ValClassifer(BLogits, BLabel, DualMOdel, ValIndex, args.save_dir,PRINT=True)
            print(' ')
            auroc = results['TMP']['AUROC']
            if auroc > best_auroc:
                best_auroc = auroc
                os.makedirs(args.save_dir,exist_ok=True)
                DetectorResume =  os.path.join(args.save_dir, 'checkpoint.tar')
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict':DualMOdel.state_dict(),
                    'best_auroc': best_auroc,
                }, DetectorResume)
                if not os.path.exists(DetectorResume):
                    print('Saving Failed!')

    clean_data, adv_data, noisy_data, label = get_data(args)
    BLogits, BLabel, CTarget, CLogits, TrainIndex, ValIndex = GetStackLogitValues(clean_data, noisy_data, adv_data,
                                                                                  label, PDmodel, FDmodel, args)
    results = ValClassifer(BLogits, BLabel, DualMOdel, ValIndex, args.save_dir,PRINT=False)
    for mtype in ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100. * results['TMP']['TNR']), end='')
    print(' {val:6.2f}'.format(val=100. * results['TMP']['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100. * results['TMP']['DTACC']), end='')
    print(' {val:6.2f}'.format(val=100. * results['TMP']['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100. * results['TMP']['AUOUT']), end='')

def TrainClassifer(BLogits,BLabel,model,criterion, optimizer, epoch,TrainIndex,SaveDir):
    model.train()
    losses = AverageMeter()
    TotalDataScale = len(TrainIndex)
    TrainIndex = TrainIndex[np.random.permutation(TotalDataScale)]
    i= 0
    open('%s/confidence_TMP_In.txt' % SaveDir, 'w').close()
    open('%s/confidence_TMP_Out.txt' % SaveDir, 'w').close()
    while not len(TrainIndex) == 0:
        if len(TrainIndex) < args.TB:
            FDx = torch.from_numpy(BLogits[1,TrainIndex[0:]].astype(dtype=np.float32))
            PDx = torch.from_numpy(BLogits[0,TrainIndex[0:]].astype(dtype=np.float32))
            target = torch.from_numpy(BLabel[TrainIndex[0:]])
        else:
            PDx = torch.from_numpy(BLogits[0,TrainIndex[0:args.TB]].astype(dtype=np.float32))
            FDx = torch.from_numpy(BLogits[1,TrainIndex[0:args.TB]].astype(dtype=np.float32))
            target = torch.from_numpy(BLabel[TrainIndex[0:args.TB]])
        target = target.cuda()
        target = target.cuda()
        PDx = torch.autograd.Variable(PDx).cuda()
        FDx = torch.autograd.Variable(FDx).cuda()
        target_var = torch.autograd.Variable(target).long()
        output = model(PDx, FDx)
        loss = criterion(output, target_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output = output.float()
        loss = loss.float()
        results = get_auroc(output,target_var,SaveDir)
        auroc = results['TMP']['AUROC']
        losses.update(loss.data, PDx.shape[0])
        if i % args.print_freq == 0:
            if (i + 1) * args.TB > TotalDataScale:
                r = '\rEpoch: [{0}][{1}/{2}] Loss {loss.avg:.4f} ' \
                    ' auroc{auroc: .3f}'.format(
                    epoch, TotalDataScale, TotalDataScale,
                    loss=losses, auroc=auroc*100)
            else:
                r = '\rEpoch: [{0}][{1}/{2}] Loss {loss.avg:.4f}' \
                    ' auroc{auroc: .3f}'.format(
                    epoch, (i + 1) * args.TB, TotalDataScale,
                    loss=losses, auroc=auroc*100)
            sys.stdout.write(r)
        i = i+1
        TrainIndex = TrainIndex[args.TB:]

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
    #### Loop according to batch size args.

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
                r = '\r[{0}/{1}]'.format(Scale, Scale )
            else:
                r = '\r[{0}/{1}]'.format((i + 1) * args.TB, Scale )
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

def get_data(args):
    try:
        clean_data = torch.load(
            '{0}/clean_data_{1}_{2}_{3}_{4}.pth'.format(args.adv_source, args.net_type, args.dataset, args.adv_type,
                                                                    args.AdvNoise))

        adv_data = torch.load(
            '{0}/adv_data_{1}_{2}_{3}_{4}.pth'.format(args.adv_source, args.net_type, args.dataset, args.adv_type,
                                                                    args.AdvNoise))

        noisy_data = torch.load(
            '{0}/noisy_data_{1}_{2}_{3}_{4}.pth'.format(args.adv_source, args.net_type, args.dataset,
                                                                      args.adv_type,
                                                                      args.AdvNoise))

        label = torch.load(
            '{0}/label_{1}_{2}_{3}_{4}.pth'.format(args.adv_source, args.net_type, args.dataset,
                                                                        args.adv_type,
                                                                        args.AdvNoise))
    except:
        print('No AEs {} !'.format(args.adv_source))

    return clean_data,adv_data, noisy_data, label

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

def GetStackLogitValues(clean_data,noisy_data,adv_data,label, PDmodel, FDmodel, args):
    Logit_Save = 'ExperimentRecord/' + args.ScriptName + '/' + args.net_type + '_' + args.dataset + '_' + \
                 args.adv_type + '_' + str(args.AdvNoise) + '/Logits'

    if not os.path.exists(Logit_Save):
        os.makedirs(Logit_Save)

    if os.path.exists(Logit_Save+'/BLogits.npy') and os.path.exists(Logit_Save+'/BLabel.npy') \
            and os.path.exists(Logit_Save+'/CTarget.npy') and os.path.exists(Logit_Save+'/CLogits.npy') and \
            os.path.exists(Logit_Save+'/Size.npy'):
        print('Load Logits from: {}'.format(Logit_Save))
        Size = np.load(Logit_Save + '/Size.npy')
        BLogits = np.load(Logit_Save+'/BLogits.npy')
        BLabel = np.load(Logit_Save+'/BLabel.npy')
        CTarget = np.load(Logit_Save+'/CTarget.npy')
        CLogits = np.load(Logit_Save+'/CLogits.npy')

    elif args.retrain:
        print('Generate Logits from source: {}_{}_{}_{}'. \
              format(args.dataset,args.net_type,args.adv_type,args.AdvNoise))
        print('Clean Data')
        CPFLogits, CPLogits, CLabel, CPF_Targets, CP_Targets,CCrtFlag = GetLogitValues(clean_data, label, PDmodel, FDmodel, args,
                                                                    Natural=True)
        INDEX = np.argmax(CPFLogits[0, :, :], 1)-np.argmax(CPFLogits[1,:,:],1)
        np.sum(INDEX)
        print('\nNoisy Data')
        NsyPFLogits, NsyPLogits, NsyLabel, NsyPF_Targets, NsyP_Targets,NCrtFlag = GetLogitValues(noisy_data, label, PDmodel,
                                                                                        FDmodel, args, Natural=True)
        INDEX = np.argmax(NsyPFLogits[0, :, :], 1)-np.argmax(NsyPFLogits[1,:,:],1)
        np.sum(INDEX)
        print('\nAdv Data')
        AdvPFLogits, AdvLabel, AdvPF_Targets,ACrtFlag = GetLogitValues(adv_data, label, PDmodel, FDmodel, args, Natural=False)

        np.size(np.nonzero(np.argmax(AdvPFLogits[0, :, :], 1) - label.numpy()))/label.shape[0]
        Size = [CPFLogits.shape[1],CPLogits.shape[1],AdvPFLogits.shape[1]]
        BLogits = np.concatenate((CPFLogits, CPLogits, NsyPFLogits, NsyPLogits, AdvPFLogits), 1)
        BLabel = np.concatenate((CLabel, NsyLabel, AdvLabel), 0)
        CTarget = np.concatenate((CPF_Targets, CP_Targets), 0)
        CLogits = np.concatenate((CPFLogits, CPLogits,), 1)
        print('\nSave Logits in: {}'.format(Logit_Save))
        np.save(Logit_Save + '/Size.npy', Size)
        np.save(Logit_Save + '/BLogits.npy', BLogits)
        np.save(Logit_Save+'/BLabel.npy',BLabel)
        np.save(Logit_Save+'/CTarget.npy',CTarget)
        np.save(Logit_Save+'/CLogits.npy',CLogits)
    else:
        print('Generate Logits from: {}_{}_{}_{}'. \
              format(args.dataset, args.net_type, args.adv_type, args.AdvNoise))
        print('Clean Data')
        CPFLogits, CPLogits, CLabel, CPF_Targets, CP_Targets, CCrtFlag = GetLogitValues(clean_data, label, PDmodel,
                                                                                        FDmodel, args,
                                                                                        Natural=True)
        print('\nNoisy Data')
        NsyPFLogits, NsyPLogits, NsyLabel, NsyPF_Targets, NsyP_Targets, NCrtFlag = GetLogitValues(noisy_data, label,
                                                                                                  PDmodel,
                                                                                                  FDmodel, args,
                                                                                                  Natural=True)
        print('\nAdv Data')
        AdvPFLogits, AdvLabel, AdvPF_Targets, ACrtFlag = GetLogitValues(adv_data, label, PDmodel, FDmodel, args,
                                                                        Natural=False)
        Size = [CPFLogits.shape[1], CPLogits.shape[1], AdvPFLogits.shape[1]]
        BLogits = np.concatenate((CPFLogits, CPLogits, NsyPFLogits, NsyPLogits, AdvPFLogits), 1)
        BLabel = np.concatenate((CLabel, NsyLabel, AdvLabel), 0)
        CTarget = np.concatenate((CPF_Targets, CP_Targets), 0)
        CLogits = np.concatenate((CPFLogits, CPLogits,), 1)
        print('\nSave Logits in: {}'.format(Logit_Save))
        np.save(Logit_Save + '/Size.npy', Size)
        np.save(Logit_Save + '/BLogits.npy', BLogits)
        np.save(Logit_Save + '/BLabel.npy', BLabel)
        np.save(Logit_Save + '/CTarget.npy', CTarget)
        np.save(Logit_Save + '/CLogits.npy', CLogits)
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


if __name__ == '__main__':
    main()
