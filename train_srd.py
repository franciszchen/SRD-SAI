import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
import os
import random
import math
import json
from collections import OrderedDict

import config_srd as config

import sys
sys.path.append("../models/")

def print_cz(str, f=None):
    if f is not None:
        print(str, file=f)
        if random.randint(0, 20) < 3:
            f.flush()
    print(str)

def time_mark():
    time_now = int(time.time())
    time_local = time.localtime(time_now)

    dt = time.strftime('%Y%m%d-%H%M%S', time_local)
    return(dt)

def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))

def model_snapshot(
    model, 
    new_file, 
    old_file=None, 
    save_dir='./', 
    verbose=True, 
    log_file=None):
    """
    :param model: network model to be saved
    :param new_file: new pth name
    :param old_file: old pth name
    :param verbose: more info or not
    :return: None
    """
    if os.path.exists(save_dir) is False:
        os.makedirs(expand_user(save_dir))
        print_cz(str='Make new dir:'+save_dir, f=log_file)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    for file in os.listdir(save_dir):
        if old_file in file:
            if verbose:
                print_cz(str="Removing old model  {}".format(expand_user(save_dir + file)), f=log_file)
            os.remove(save_dir + file) # 先remove旧的pth，再存储新的
    if verbose:
        print_cz(str="Saving new model to {}".format(expand_user(save_dir + new_file)), f=log_file)
    torch.save(model, expand_user(save_dir + new_file))

def prepare():
    """
        config, make dirs and logger file
    """
    args = config.get_args()
    time_tag = time_mark()
    log_dir = config.save_dir_docker + time_tag + '_' + args.theme + '_shuffle_'+args.cross_str+ '_dualr'+ str(args.dual_ratio)+ '_regular'+str(args.regular_ratio)+'_rank'+str(args.rank_ratio)+ '_ensemble'+str(args.ensemble_ce_ratio)+'_super'+str(args.super_ce_ratio)+ '_' + args.optim + '_lr' + str(args.lr)+'_lrsr' + str(args.lr_factor_srnet) + '_wd'+str(args.weight_decay) +\
        '_bs'+str(args.batch_size)+'_epochs'+str(args.epochs) + '_interval'+str(args.lr_decay_interval) +'_gamma'+str(args.lr_decay_gamma)

    if os.path.exists(log_dir) is False:# make dir if not exist
        os.makedirs(expand_user(log_dir))
        print('make dir: ' + str(log_dir))
    return args,  log_dir

def adjust_learning_rate(optimizer, optimizer_srnet, lr, lr_factor_srnet, epoch, lr_decay_interval=40, lr_decay_gamma=0.1):
    """Sets the learning rate to the initial LR"""
    lr = lr * (lr_decay_gamma ** (epoch // lr_decay_interval)) 
    ###
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    ###
    for param_group in optimizer_srnet.param_groups:
        param_group['lr'] = lr*lr_factor_srnet

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0# value = current value
        self.avg = 0
        self.sum = 0# weighted sum
        self.count = 0# total sample num

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # output, target -> FloatTensor, LongTensor
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)# maxk preds
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))# expand target

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))# 
    return res

def train(
    train_loader, 
    test_loader, 
    model, 
    criterion, 
    criterion_dual_l2s, 
    criterion_dual_s2l, 
    criterion_regularization, 
    optimizer, 
    optimizer_srnet,
    lr, 
    lr_factor_srnet, 
    epochs, 
    batch_size, 
    lr_decay_interval=1, 
    lr_decay_gamma=0.1, 
    logfile=None, 
    save_path=None, 
    start_epoch=0, 
    pth_prefix='', 
    csv_flag = False, 
    dual_ratio=1, 
    regular_ratio=1, 
    rank_ratio=1, 
    ensemble_ce_ratio=1.0, 
    super_ce_ratio=1.0, 
    low_ce_ratio=1.0, 
    aux_ce_ratio=1.0
    ):
    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, optimizer_srnet, lr, lr_factor_srnet, epoch, lr_decay_interval, lr_decay_gamma)
        print_cz(str=' ==> Train Epoch:\t{:d}\t lr:{:.3e} \t lr_srnet:{:.3e}'.format(
            epoch, optimizer.param_groups[0]['lr'], optimizer_srnet.param_groups[0]['lr']), f=logfile)
        
        train_top1_ensemble, train_loss_total = train_a_epoch(
                train_loader=train_loader, 
                model=model, 
                criterion=criterion, 
                criterion_dual_l2s=criterion_dual_l2s,
                criterion_dual_s2l=criterion_dual_s2l, 
                criterion_regularization=criterion_regularization, 
                optimizer=optimizer, 
                optimizer_srnet=optimizer_srnet,
                epoch=epoch, 
                batch_size=batch_size, 
                dual_ratio=dual_ratio, 
                regular_ratio=regular_ratio, 
                rank_ratio=rank_ratio, 
                ensemble_ce_ratio=ensemble_ce_ratio, 
                super_ce_ratio=super_ce_ratio, 
                low_ce_ratio=low_ce_ratio, 
                aux_ce_ratio=aux_ce_ratio, 
                logfile=logfile
            )
        
        print_cz(" ==> Test ", f=logfile)
        test_top1_ensemble, test_loss_total = test(
                model, 
                test_loader=test_loader, 
                criterion=criterion, 
                criterion_dual_l2s=criterion_dual_l2s, 
                criterion_dual_s2l=criterion_dual_s2l, 
                criterion_regularization=criterion_regularization, 
                batch_size=batch_size, 
                epoch=epoch, 
                dual_ratio=dual_ratio, 
                regular_ratio=regular_ratio, 
                rank_ratio=rank_ratio,
                ensemble_ce_ratio=ensemble_ce_ratio, 
                super_ce_ratio=super_ce_ratio, 
                low_ce_ratio=low_ce_ratio, 
                aux_ce_ratio=aux_ce_ratio,
                logfile=logfile
            )
        # save model periodically
        if epoch % 20 == 9:
            model_snapshot(model, new_file=(
                    pth_prefix+'model-period-{}-acc{:.3f}-{}.pth'.format(epoch,
                                                 test_top1_ensemble, time_mark())
                    ), old_file=pth_prefix + 'model-no-remove-', save_dir=save_path , verbose=True)
        # save model at the end of training
        if epoch + 10 > epochs:
            model_snapshot(model, new_file=(
                    pth_prefix+'model-last-{}-acc{:.3f}-super{:.3f}-low{:.3f}-aux{:.3f}-{}.pth'.format(epoch, test_top1_ensemble, time_mark())
                    ), old_file=pth_prefix + 'model-no-remove-', save_dir=save_path , verbose=True)

def train_a_epoch(
    train_loader, 
    model, 
    criterion, 
    criterion_dual_l2s, 
    criterion_dual_s2l, 
    criterion_regularization, 
    optimizer, 
    optimizer_srnet, 
    epoch, 
    batch_size,
    dual_ratio=1, 
    regular_ratio=1, 
    rank_ratio=1, 
    ensemble_ce_ratio=1.0, 
    super_ce_ratio=1.0, 
    low_ce_ratio=1.0, 
    aux_ce_ratio=1.0,
    logfile=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_ce_ensemble = AverageMeter()
    losses_ce_aux = AverageMeter()
    losses_ce_low = AverageMeter()
    losses_ce_super = AverageMeter()
    losses_dual_l2s = AverageMeter()
    losses_dual_s2l = AverageMeter()
    losses_regular = AverageMeter()
    losses_rank = AverageMeter()
    losses_total = AverageMeter()
    top1_ensemble = AverageMeter()

    model.cuda()
    model.train()
    epoch_start_time = time.time()
    end = time.time()

    for idx, (img, label) in enumerate(train_loader):

        model.zero_grad()
        optimizer.zero_grad()
        optimizer_srnet.zero_grad()
        input_batch = Variable(img).cuda()
        label_batch = Variable(label).long().cuda()
        # data time
        data_time.update(time.time() - end)
        output_low, output_super, output_aux, output_ensemble, M_s2l_list, M_l2s_list, A_low_list, A_super_list  = model(x_low=input_batch)
        # rank loss
        logit_list = [output_low, output_super, output_aux, output_ensemble]
        preds = []
        for i in range(label_batch.shape[0]): 
            pred = [logit[i][label_batch[i]] for logit in logit_list]
            preds.append(pred)
        loss_rank = rank_loss.pairwise_ranking_loss(preds)
        loss_ce_ensemble = criterion(output_ensemble, label_batch)
        loss_ce_aux = criterion(output_aux, label_batch)
        loss_ce_low = criterion(output_low, label_batch)
        loss_ce_super = criterion(output_super, label_batch)
        loss_dual_s2l = criterion_dual_s2l(f_var=A_super_list[0], f_fix=A_low_list[0]) + criterion_dual_s2l(f_var=A_super_list[1], f_fix=A_low_list[1]) +\
                        criterion_dual_s2l(f_var=A_super_list[2], f_fix=A_low_list[2]) + criterion_dual_s2l(f_var=A_super_list[3], f_fix=A_low_list[3])
        loss_dual_l2s = criterion_dual_l2s(f_var=A_low_list[0], f_fix=A_super_list[0]) + criterion_dual_l2s(f_var=A_low_list[1], f_fix=A_super_list[1]) +\
                        criterion_dual_l2s(f_var=A_low_list[2], f_fix=A_super_list[2]) + criterion_dual_l2s(f_var=A_low_list[3], f_fix=A_super_list[3])
        loss_regularization = 0.5*(
                criterion_regularization(f_var=M_s2l_list[3], f_fix=M_s2l_list[2]) + criterion_regularization(f_var=M_s2l_list[2], f_fix=M_s2l_list[1]) +criterion_regularization(f_var=M_s2l_list[1], f_fix=M_s2l_list[0])
                + criterion_regularization(f_var=M_l2s_list[3], f_fix=M_l2s_list[2]) + criterion_regularization(f_var=M_l2s_list[2], f_fix=M_l2s_list[1]) +criterion_regularization(f_var=M_l2s_list[1], f_fix=M_l2s_list[0])
                )
                                                  

        loss_total = ensemble_ce_ratio*loss_ce_ensemble + aux_ce_ratio*loss_ce_aux + low_ce_ratio*loss_ce_low + super_ce_ratio*loss_ce_super +\
            (dual_ratio*(loss_dual_s2l+loss_dual_l2s)/2.0) + regular_ratio*loss_regularization +\
            rank_ratio*loss_rank

        prec1_ensemble = accuracy(output_ensemble.data, label_batch.data, topk=(1,))[0]

        losses_ce_ensemble.update(ensemble_ce_ratio*loss_ce_ensemble.data.item(), input_batch.shape[0])
        losses_ce_aux.update(aux_ce_ratio*loss_ce_aux.data.item(), input_batch.shape[0])
        losses_ce_low.update(low_ce_ratio*loss_ce_low.data.item(), input_batch.shape[0])
        losses_ce_super.update(super_ce_ratio*loss_ce_super.data.item(), input_batch.shape[0])
        losses_dual_l2s.update(dual_ratio*loss_dual_l2s.data.item(), input_batch.shape[0]) 
        losses_dual_s2l.update(dual_ratio*loss_dual_s2l.data.item(), input_batch.shape[0]) # update dual_s2l
        losses_regular.update(regular_ratio*loss_regularization.data.item(), input_batch.shape[0])
        losses_rank.update(rank_ratio*loss_rank.data.item(), input_batch.shape[0])
        losses_total.update(loss_total.data.item(), input_batch.shape[0])

        top1_ensemble.update(prec1_ensemble[0], input_batch.shape[0])

        model.zero_grad()
        optimizer.zero_grad()
        optimizer_srnet.zero_grad()
        loss_total.backward()
        optimizer.step()
        optimizer_srnet.step()

        # batch time updated
        batch_time.update(time.time() - end)
        end = time.time()

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time

    print_cz(str=' * Train time {:.3f}  BatchT:{:.3f}  DataT:{:.3f}  D/B:{:.1f}%'.format(epoch_time, batch_time.avg, data_time.avg, 100.0*(data_time.avg/batch_time.avg)), f=logfile)
    print_cz(str='   Total_Loss {:.3f}  Ensemble_Loss:{:.3f}  Super_Loss:{:.3f}  Low_Loss:{:.3f}  Aux_Loss:{:.3f}  Dual_Loss_l2s:{:.3f}  Dual_Loss_s2l:{:.3f}  Regular_Loss:{:.3f}  Rank_Loss:{:.3f}'\
        .format(losses_total.avg, losses_ce_ensemble.avg, losses_ce_super.avg, losses_ce_low.avg, losses_ce_aux.avg, losses_dual_l2s.avg, losses_dual_s2l.avg, losses_regular.avg, losses_rank.avg), f=logfile)
    print_cz(str='   Prec@1: {:.3f}%'.format(top1_ensemble.avg), f=logfile)
    return top1_ensemble.avg, losses_total.avg


def test(
    model, 
    test_loader, 
    criterion, 
    criterion_dual_l2s, 
    criterion_dual_s2l, 
    criterion_regularization, 
    batch_size, 
    epoch=0, 
    dual_ratio=1, 
    regular_ratio=1, 
    rank_ratio=1, 
    ensemble_ce_ratio=1.0, 
    super_ce_ratio=1.0, 
    low_ce_ratio=1.0, 
    aux_ce_ratio=1.0,
    logfile=None
    ):

    model.cuda()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_ce_ensemble = AverageMeter()
    losses_ce_aux = AverageMeter()
    losses_ce_low = AverageMeter()
    losses_ce_super = AverageMeter()
    losses_dual_l2s = AverageMeter()
    losses_dual_s2l = AverageMeter()
    losses_regular = AverageMeter()
    losses_rank = AverageMeter()
    losses_total = AverageMeter()
    top1_ensemble = AverageMeter()
    end = time.time()
    test_start_time = time.time()

    with torch.no_grad():
        for idx, (img, label) in enumerate(test_loader):
            input_batch = Variable(img, volatile=True).cuda()
            label_batch = torch.autograd.Variable(label, volatile=True).long().cuda()
            # data time
            data_time.update(time.time() - end)
            # compute output
            output_low, output_super, output_aux, output_ensemble, M_s2l_list, M_l2s_list, A_low_list, A_super_list = model(x_low=input_batch)
            
            logit_list = [output_low, output_super, output_aux, output_ensemble]
            preds = []
            for i in range(label_batch.shape[0]): 
                pred = [logit[i][label_batch[i]] for logit in logit_list]
                preds.append(pred)
            loss_rank = rank_loss.pairwise_ranking_loss(preds)

            loss_ce_ensemble = criterion(output_ensemble, label_batch)
            loss_ce_aux = criterion(output_aux, label_batch)
            loss_ce_low = criterion(output_low, label_batch)
            loss_ce_super = criterion(output_super, label_batch)
            loss_dual_s2l = criterion_dual_s2l(f_var=A_super_list[0], f_fix=A_low_list[0]) + criterion_dual_s2l(f_var=A_super_list[1], f_fix=A_low_list[1]) +\
                        criterion_dual_s2l(f_var=A_super_list[2], f_fix=A_low_list[2]) + criterion_dual_s2l(f_var=A_super_list[3], f_fix=A_low_list[3])
            loss_dual_l2s = criterion_dual_l2s(f_var=A_low_list[0], f_fix=A_super_list[0]) + criterion_dual_l2s(f_var=A_low_list[1], f_fix=A_super_list[1]) +\
                        criterion_dual_l2s(f_var=A_low_list[2], f_fix=A_super_list[2]) + criterion_dual_l2s(f_var=A_low_list[3], f_fix=A_super_list[3])
            loss_regularization = 0.5*(
                criterion_regularization(f_var=M_s2l_list[3], f_fix=M_s2l_list[2]) + criterion_regularization(f_var=M_s2l_list[2], f_fix=M_s2l_list[1]) + criterion_regularization(f_var=M_s2l_list[1], f_fix=M_s2l_list[0])
                +criterion_regularization(f_var=M_l2s_list[3], f_fix=M_l2s_list[2]) + criterion_regularization(f_var=M_l2s_list[2], f_fix=M_l2s_list[1]) + criterion_regularization(f_var=M_l2s_list[1], f_fix=M_l2s_list[0])
                )

            loss_total = ensemble_ce_ratio*loss_ce_ensemble + aux_ce_ratio*loss_ce_aux+ low_ce_ratio*loss_ce_low + super_ce_ratio*loss_ce_super +\
                    dual_ratio*(loss_dual_l2s+loss_dual_s2l)/2.0 + regular_ratio*loss_regularization +\
                    rank_ratio*loss_rank

            prec1_ensemble = accuracy(output_ensemble.data, label_batch.data, topk=(1,))[0]

            losses_ce_ensemble.update(ensemble_ce_ratio*loss_ce_ensemble.data.item(), input_batch.shape[0])
            losses_ce_aux.update(aux_ce_ratio*loss_ce_aux.data.item(), input_batch.shape[0])
            losses_ce_low.update(low_ce_ratio*loss_ce_low.data.item(), input_batch.shape[0])
            losses_ce_super.update(super_ce_ratio*loss_ce_super.data.item(), input_batch.shape[0])
            losses_dual_l2s.update(dual_ratio*loss_dual_l2s.data.item(), input_batch.shape[0])
            losses_dual_s2l.update(dual_ratio*loss_dual_s2l.data.item(), input_batch.shape[0])
            losses_regular.update(regular_ratio*loss_regularization.data.item(), input_batch.shape[0])
            losses_rank.update(rank_ratio*loss_rank.data.item(), input_batch.shape[0])
            losses_total.update(loss_total.data.item(), input_batch.shape[0])
            top1_ensemble.update(prec1_ensemble[0], input_batch.shape[0])
            # batch time updated
            batch_time.update(time.time() - end)
            end = time.time()

    test_end_time = time.time()
    test_time = test_end_time - test_start_time

    print_cz(str=' * Test  time {:.3f}  BatchT:{:.3f}  DataT:{:.3f}  D/B:{:.1f}%'.format(test_time, batch_time.avg, data_time.avg, 100.0*(data_time.avg/batch_time.avg)), f=logfile)
    print_cz(str='   Total_Loss {:.3f}  Ensemble_Loss:{:.3f}  Super_Loss:{:.3f}  Low_Loss:{:.3f}  Aux_Loss:{:.3f}  Dual_Loss_l2s:{:.3f}  Dual_Loss_s2l:{:.3f}  Regular_Loss:{:.3f}  Rank_Loss:{:.3f}'.format(losses_total.avg, losses_ce_ensemble.avg, losses_ce_super.avg, losses_ce_low.avg, losses_ce_aux.avg, losses_dual_l2s.avg, losses_dual_s2l.avg, losses_regular.avg, losses_rank.avg), f=logfile)
    print_cz(str='   Prec@1: {:.3f}%'.format(top1_ensemble.avg), f=logfile)
    return top1_ensemble.avg, losses_total.avg 


if __name__ == '__main__':
    from models import model_srd, mrcnet 
    from data import dataset_srd
    from loss import dual_loss, regularization, rank_loss

    print('start...')
    args, log_dir = prepare()
    log_file = open((log_dir + '/' + 'print_out_screen.txt'), 'w')
    print_cz("===> Preparing", f=log_file)
    t = time.time()
    with open(log_dir + '/setting.json', 'w') as f:
        f.write(json.dumps(args.__dict__, indent=4))
        print_cz(json.dumps(args.__dict__, indent=4), f=log_file)

    print_cz("===> Building model", f=log_file)
    sr_model = mrcnet.Net_Student(interpolate='bicubic')
    sr_model.load_state_dict(torch.load('sr pre-trained pth'))
    model_orig = model_srd.DualNet18_18(sr_net=sr_model)
    model_orig_dict = model_orig.state_dict()
    pretrained_low_dict = torch.load('pre-trained resnet-18 for lr branch')
    pretrained_super_dict = torch.load('pre-trained resnet-18 for sr branch')
    model_orig_dict.update(pretrained_low_dict)
    model_orig_dict.update(pretrained_super_dict)
    model_orig.load_state_dict(model_orig_dict)

    criterion = nn.CrossEntropyLoss()
    criterion_dual_l2s = dual_loss.DualUniform_2Optim_Fnorm()
    criterion_dual_s2l = dual_loss.DualUniform_2Optim_Fnorm()
    criterion_regularization = regularization.Regularization()

    print_cz("===> Setting GPU", f=log_file)
    if args.job_type == 'S' or args.job_type == 's':
        model = model_orig.cuda()
    else:
        if args.job_type == 'Q' or args.job_type == 'q':
            gpu_device_ids=[0, 1, 2, 3]
        elif args.job_type == 'E' or args.job_type == 'e':
            gpu_device_ids=[0, 1, 2, 3, 4, 5, 6, 7]
        elif args.job_type == 'D' or args.job_type == 'd':
            gpu_device_ids=[0, 1]
        model = nn.DataParallel(model_orig.cuda(), device_ids=gpu_device_ids).cuda()

    print_cz("===> Loading datasets", f=log_file)
    train_loader = dataset_srd.get_loader(batch_size=args.batch_size, cross_str=args.cross_str, stage='train', size=args.resolution, num_workers=args.num_workers, augmentations=True)
    test_loader = dataset_srd.get_loader(batch_size=args.batch_size, cross_str=args.cross_str, stage='test', size=args.resolution, num_workers=args.num_workers, augmentations=False)

    print_cz("===> Setting Optimizer", f=log_file)
    count = 0
    for name in [name for name, _ in model.named_parameters() if ('ssi_' in name or 'bottle_' in name or 'gate_' in name)]:
        count+=1
    print(len([name for name, _ in model.named_parameters() if ('ssi_' in name or 'bottle_' in name or 'gate_' in name)]))
    if args.optim in ['Adam', 'adam']:
        optimizer = torch.optim.Adam(params=[param for name, param in model.named_parameters() 
            if (
                ('ssi_' in name) or 
                ('bottle_' in name) or 
                ('gate_' in name) or 
                ('ensemble_layer' in name)  or
                ('super' in name) or 
                ('low' in name)
                )
            ], lr=args.lr, weight_decay=args.weight_decay)
        optimizer_srnet = torch.optim.Adam(params=[param for name, param in model.named_parameters() 
            if ('sr_net' in name)], 
            lr=args.lr*args.lr_factor_srnet,
            weight_decay=args.weight_decay)
    elif args.optim in ['SGD', 'sgd']:
        optimizer = torch.optim.SGD(params=[param for name, param in model.named_parameters() 
            if (
                ('ssi_' in name) or 
                ('bottle_' in name) or 
                ('gate_' in name) or 
                ('ensemble_layer' in name)  or
                ('super' in name) or 
                ('low' in name)
                )
            ], lr=args.lr, weight_decay=args.weight_decay, momentum=args.momen)
        optimizer_srnet = torch.optim.SGD(
            params=[param for name, param in model.named_parameters() 
            if ('sr_net' in name)], 
            lr=args.lr*args.lr_factor_srnet, 
            weight_decay=args.weight_decay, 
            momentum=args.momen)
        
    print_cz("===> Training", f=log_file)
    train(
        train_loader=train_loader, 
        test_loader=test_loader, 
        model=model, 
        criterion=criterion, 
        criterion_dual_l2s=criterion_dual_l2s, 
        criterion_dual_s2l=criterion_dual_s2l, 
        criterion_regularization=criterion_regularization, 
        optimizer=optimizer, 
        optimizer_srnet=optimizer_srnet, 
        lr=args.lr, 
        lr_factor_srnet=args.lr_factor_srnet,
        epochs=args.epochs, 
        batch_size=args.batch_size,  
        lr_decay_interval=args.lr_decay_interval, 
        lr_decay_gamma=args.lr_decay_gamma, 
        logfile=log_file, 
        save_path=log_dir+'/', 
        dual_ratio=args.dual_ratio, 
        regular_ratio=args.regular_ratio, 
        rank_ratio=args.rank_ratio, 
        ensemble_ce_ratio=args.ensemble_ce_ratio, 
        super_ce_ratio=args.super_ce_ratio, 
        low_ce_ratio=args.low_ce_ratio, 
        aux_ce_ratio=args.aux_ce_ratio
    )

    print_cz(str(time.time()-t), f=log_file)
    log_file.close()

