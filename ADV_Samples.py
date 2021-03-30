from __future__ import print_function
import warnings
from ModelData import *
warnings.filterwarnings('ignore')
import argparse
import torch
import torch.nn as nn
import data_loader
import os
import lib.adversary as adversary
from get_model import get_model, get_dual_model
from torchvision import transforms
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=False, help='cifar10 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', required=False, help='resnet | densenet')
parser.add_argument('--gpuid', type=str, default=0, help='gpu index')
parser.add_argument('--adv_type', required=False, help='FGSM | BIM | DeepFool | CWL2')
parser.add_argument('--domain', type=str)
parser.add_argument('--wave', type=str)
parser.add_argument('--AdvNoise', type=float)
parser.add_argument('--adv_parameter', type=float,help='Parameter to control perturbation magnitude')
parser.set_defaults(wave = 'sym17')
parser.set_defaults(gpuid = '4')
parser.set_defaults(net_type='resnet')
parser.set_defaults(dataset='cifar10')
parser.set_defaults(domain='PD')
parser.set_defaults(adv_type='FGSM')
parser.set_defaults(adv_parameter=0.1)


def main():
    global args
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    args.outf = 'adv_output/' + args.net_type + '_' + args.dataset + '_' + args.adv_type +'/'
    os.makedirs(args.outf,exist_ok=True)
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)
    torch.cuda.manual_seed(0)
    args.num_classes,min_pixel,max_pixel,random_noise_size = Get_Parameters(args)
    model = GET_MODEL(args,args.net_type)
    args.domain = 'FD'
    FDmodel = GET_MODEL(args,args.net_type)
    model.cuda()
    FDmodel.cuda()

    _, test_loader = GET_DATA(args)
    adv_data_tot, clean_data_tot, noisy_data_tot = 0, 0, 0
    label_tot = 0

    FDcorrect, correct, FDadv_correct, adv_correct, noise_correct = 0, 0, 0, 0, 0
    total, generated_noise = 0, 0

    criterion = nn.CrossEntropyLoss().cuda()

    selected_list = []
    selected_index = 0
    count = 0

    for data, target in test_loader:
        FDmodel.eval()
        model.eval()
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        FDoutput = FDmodel(data)
        # compute the accuracy
        pred = output.data.max(1)[1]
        FDpred = FDoutput.data.max(1)[1]
        equal_flag = pred.eq(target.data).cpu()
        FDequal_flag = FDpred.eq(target.data).cpu()
        correct += equal_flag.sum()
        FDcorrect += FDequal_flag.sum()
        noisy_data = torch.add(data.data, random_noise_size, torch.randn(data.size()).cuda())
        noisy_data = torch.clamp(noisy_data, min_pixel, max_pixel)
        if total == 0:
            clean_data_tot = data.clone().data.cpu()
            label_tot = target.clone().data.cpu()
            noisy_data_tot = noisy_data.clone().cpu()
        else:
            clean_data_tot = torch.cat((clean_data_tot, data.clone().data.cpu()),0)
            label_tot = torch.cat((label_tot, target.clone().data.cpu()), 0)
            noisy_data_tot = torch.cat((noisy_data_tot, noisy_data.clone().cpu()),0)
        # generate adversarial
        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True)
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        # attacking
        adv_data = Attack(args, data, model, criterion, target, inputs, args.adv_parameter, min_pixel, max_pixel)
        # measure the noise
        temp_noise_max = torch.abs((data.data - adv_data).view(adv_data.size(0), -1))
        temp_noise_max = torch.norm(temp_noise_max, 2)
        generated_noise += temp_noise_max


        if total == 0:
            flag = 1
            adv_data_tot = adv_data.clone().cpu()
        else:
            adv_data_tot = torch.cat((adv_data_tot, adv_data.clone().cpu()),0)

        output = model(Variable(adv_data, volatile=True))
        FDoutput = FDmodel(Variable(adv_data, volatile=True))
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag_adv = pred.eq(target.data).cpu()
        adv_correct += equal_flag_adv.sum()
        FDpred = FDoutput.data.max(1)[1]
        FDequal_flag_adv = FDpred.eq(target.data).cpu()
        FDadv_correct += FDequal_flag_adv.sum()
        output = model(Variable(noisy_data, volatile=True))
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag_noise = pred.eq(target.data).cpu()
        noise_correct += equal_flag_noise.sum()
        TempSelectedIndex = []
        for i in range(data.size(0)):
            if equal_flag[i] == 1 and equal_flag_noise[i] == 1 and equal_flag_adv[i] == 0:
                selected_list.append(selected_index)
                TempSelectedIndex.append(i)
            selected_index += 1

        SuccAdv = adv_data[TempSelectedIndex]
        SuccImg = data.data[TempSelectedIndex]
        SuccLab = target[TempSelectedIndex]
        pred = model(Variable(SuccImg, volatile=True)).data.max(1)[1]
        Advpred = model(Variable(SuccAdv, volatile=True)).data.max(1)[1]
        equal_flag_adv = Advpred.eq(SuccLab.data).cpu()
        equal_flag = pred.eq(SuccLab.data).cpu()
        FDpred = FDmodel(Variable(SuccImg, volatile=True)).data.max(1)[1]
        AdvFDpred = FDmodel(Variable(SuccAdv, volatile=True)).data.max(1)[1]
        FDequal_flag = FDpred.eq(SuccLab.data).cpu()
        FDequal_flag_adv = AdvFDpred.eq(SuccLab.data).cpu()
        total += data.size(0)
        AdvNoise = temp_noise_max/data.size(0)
        BatchAc = torch.sum(equal_flag).numpy()/equal_flag.shape[0]
        AdvAc = torch.sum(equal_flag_adv).numpy()/equal_flag.shape[0]

        FDAdvAc = torch.sum(FDequal_flag_adv).numpy() / equal_flag.shape[0]
        FDAc = torch.sum(FDequal_flag).numpy()/equal_flag.shape[0]
        print('\t [{4}/{5}] Adv Noise: {0:.2f}; Final Accuracy: {1:.2f}; Adv Accuracy: {2:.2f}; FDAdv Accuracy: {6:.2f}; FD Accuracy: {3:.2f}'.format(
            AdvNoise, BatchAc, AdvAc, FDAc,count,len(test_loader),FDAdvAc))
        count += 1
        break
    selected_list = torch.LongTensor(selected_list)
    clean_data_tot = torch.index_select(clean_data_tot, 0, selected_list)
    adv_data_tot = torch.index_select(adv_data_tot, 0, selected_list)
    noisy_data_tot = torch.index_select(noisy_data_tot, 0, selected_list)
    label_tot = torch.index_select(label_tot, 0, selected_list)
    AdvNoise = generated_noise / total
    Adv_sample_dir = os.path.join(args.outf,'{:.2f}_{:.2f}'.format(AdvNoise,args.adv_parameter))
    print(Adv_sample_dir)
    os.makedirs(Adv_sample_dir,exist_ok=True)
    np.save(Adv_sample_dir + '/clean_data_{0}_{1}_{2}_{3:.2f}.npy'.format(args.net_type, args.dataset, args.adv_type, AdvNoise),clean_data_tot.numpy())
    np.save(Adv_sample_dir + '/adv_data_{0}_{1}_{2}_{3:.2f}.npy'.format(args.net_type, args.dataset, args.adv_type, AdvNoise),adv_data_tot.numpy())
    np.save(Adv_sample_dir + '/noisy_data_{0}_{1}_{2}_{3:.2f}.npy'.format(args.net_type, args.dataset, args.adv_type, AdvNoise),noisy_data_tot.numpy())
    np.save(Adv_sample_dir + '/label_{0}_{1}_{2}_{3:.2f}.npy'.format(args.net_type, args.dataset, args.adv_type, AdvNoise),label_tot.numpy())
    print('Adversarial Noise:({:.2f})\n'.format(generated_noise / total))
    print('Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * correct / total))
    print('Final FD Accuracy: {}/{} ({:.2f}%)\n'.format(FDcorrect, total, 100. * FDcorrect / total))
    print('Adversarial Accuracy: {}/{} ({:.2f}%)\n'.format(adv_correct, total, 100. * adv_correct / total))
    print('FD Adversarial Accuracy: {}/{} ({:.2f}%)\n'.format(FDadv_correct, total, 100. * FDadv_correct / total))
    print('Noisy Accuracy: {}/{} ({:.2f}%)\n'.format(noise_correct, total, 100. * noise_correct / total))
    print('Model: {}; Data: {}; Adv: {}_{}'.format(args.net_type, args.dataset, args.adv_type, args.adv_parameter))

def GET_DATA(args):
    if args.net_type == 'VggBn':
        in_transform = transforms.Compose([transforms.ToTensor(), \
                                       transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), \
                                                            (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)), ])
    elif args.net_type == 'resnet':
        in_transform = transforms.Compose([transforms.ToTensor(), \
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    if not args.dataset == 'imagenet':
        train_loader, val_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)
    else:
        val_info = './data/ImageLog.txt'
        train_loader = []
        val_loader = Mydataloader([122, 122], val_info, args.batch_size, 0,mode='pixel',shuffle=True)

    return train_loader, val_loader

def Get_Parameters(args):
    #### min_pixel, max_pixel
    if args.net_type == 'densenet':
        min_pixel = -1.98888885975
        max_pixel = 2.12560367584
    if args.net_type == 'resnet':
        min_pixel = -2.72906570435
        max_pixel = 2.95373125076
    if args.net_type == 'VggBn':
        min_pixel = -2.72906570435
        max_pixel = 2.95373125076
    #### noise_size
    if args.dataset == 'cifar10':
        num_class = 10
        if args.adv_type == 'FGSM':
            random_noise_size = 0.21 / 4
        elif args.adv_type == 'BIM':
            random_noise_size = 0.21 / 4
        elif args.adv_type == 'DeepFool':
            random_noise_size = 0.13 * 2 / 10
        elif args.adv_type == 'CWL2':
            random_noise_size = 0.03 / 2
    elif args.dataset == 'imagenet':
        num_class = 100
        if args.adv_type == 'FGSM':
            random_noise_size = 0.21 / 8
        elif args.adv_type == 'BIM':
            random_noise_size = 0.21 / 8
        elif args.adv_type == 'DeepFool':
            random_noise_size = 0.13 * 2 / 8
        elif args.adv_type == 'CWL2':
            random_noise_size = 0.06 / 5
    elif args.dataset == 'svhn':
        num_class = 10
        if args.adv_type == 'FGSM':
            random_noise_size = 0.21 / 4
        elif args.adv_type == 'BIM':
            random_noise_size = 0.21 / 4
        elif args.adv_type == 'DeepFool':
            random_noise_size = 0.16 * 2 / 5
        elif args.adv_type == 'CWL2':
            random_noise_size = 0.07 / 2
    return num_class,min_pixel,max_pixel,random_noise_size

def GET_MODEL(args,net_type):
    args.resume = './pre_trained/' + args.domain+ net_type + '_' + args.dataset + '.pth'
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

def Attack(args,data,model,criterion,target,inputs,adv_noise,min_pixel, max_pixel):

    if args.adv_type == 'FGSM':
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if args.net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
        else:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

    elif args.adv_type == 'BIM':
        gradient = torch.sign(inputs.grad.data)
        for k in range(5):
            inputs = torch.add(inputs.data, adv_noise, gradient)
            inputs = torch.clamp(inputs, min_pixel, max_pixel)
            inputs = Variable(inputs, requires_grad=True)
            output = model(inputs)
            loss = criterion(output, target)
            loss.backward()
            gradient = torch.sign(inputs.grad.data)
            if args.net_type == 'densenet':
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
            else:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

    if args.adv_type == 'DeepFool':

        _, adv_data = adversary.deepfool(model, data.data.clone(), target.data.cpu(), \
                                         args.num_classes, step_size=adv_noise, train_mode=False)
        adv_data = adv_data.cuda()

    elif args.adv_type == 'CWL2':
        _, adv_data = adversary.cw(model, data.data.clone(), target.data.cpu(), 1.0, 'l2', crop_frac=1.0)
    else:
        adv_data = torch.add(inputs.data, adv_noise, gradient)

    adv_data = torch.clamp(adv_data, min_pixel, max_pixel)
    return adv_data

if __name__ == '__main__':
    main()