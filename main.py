'''Train CIFAR10 with PyTorch.'''
import functools
from numpy import full
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import optimizer

import torchvision
import torchvision.transforms as transforms

import os
import random
import copy
import argparse

from models import *
from utils import progress_bar
from cl import EvalProgressPerSampleClassification as EPSP, \
    FixDataMemoryBatchClassification as FD, \
    MetricClassification as MC, \
    ClassificationTrain as VT, \
    ClassificationMask as CM, \
    FastGradientSign as FGS,\
    AvgNet
from cl.configs.imageclass_config import incremental_config
from cl.utils import PytorchModeWrap, get_config, get_config_default, save_config, set_config, repeat_dataloader
from cl.algo import knowledge_distill_loss, EWC
from cl.algo.torchensemble import SnapshotEnsembleClassifier, evaluate_uncertainty, evaluate_uncertainty_part, BaggingClassifier,  FastGeometricClassifier, evaluate_consistency
from torch.utils.data import ConcatDataset
from cl.dataset.imagenet32 import Imagenet32
import cl
from functools import partial
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--interpolate', action='store_true')
parser.add_argument("--dataset", default="cifar10")
parser.add_argument("--smalldata",action="store_true")
parser.add_argument("--mix-label", default="",choices=["avg", "","adpt_avg"])
parser.add_argument("--unsupdata",default="")
parser.add_argument("--ensemble", default="")
parser.add_argument("--segment",default=2, type=int)
parser.add_argument("--hist-avg",action="store_true")
parser.add_argument("--trainaug",default="CF")
parser.add_argument("--occulusion", action="store_true")
parser.add_argument("--data-enlarge", action="store_true")
parser.add_argument("--dist-weight",default="")
parser.add_argument("--improve-loss",action="store_true")
parser.add_argument("--improve-loss-beta",default=0.5,type=float)
parser.add_argument("--unsup-kd",action="store_true")
parser.add_argument("--consistent-improve", action="store_true")
parser.add_argument("--net", default="ResNet18v2")
parser.add_argument("--lwf", action="store_true")
parser.add_argument("--loss",default="xent", choices=["l1","xent", "l1_xent"])
parser.add_argument("--lwf-lambda", default=1.0, type=float)
parser.add_argument("--scratch", action="store_true")
parser.add_argument("--ewc", action="store_true")
parser.add_argument("--ensemble-num",default=5,type=int)
parser.add_argument("--ensemble-fast",action="store_true")
parser.add_argument("--ensemble-best",action="store_true")
parser.add_argument("--ewc-lambda", default=5000.0)
parser.add_argument("--max-epoch", default=200,type=int)
parser.add_argument("--dev-scene", default="sequential")
parser.add_argument("--inc-setting", default="data_inc")
parser.add_argument("--warmup-ep",default = 0, type=int)
parser.add_argument("--class-seed", default=0)
parser.add_argument("--model-update",action="store_true")
parser.add_argument("--kd-model", default = "")
parser.add_argument("--correct-set", action="store_true")
parser.add_argument("--var-kd",default="[1]")
parser.add_argument("--seploss", action="store_true")
parser.add_argument("--train-ext",action="store_true")
parser.add_argument("--opt",default="sgd")
parser.add_argument("--skip-exist",action="store_true")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
assert args.trainaug in ["", "CF", "ADV","CF_ADV"]
assert args.dataset in ["cifar10", "cifar100", "imagenet32"]
assert args.kd_model in ["", "pretrain"]
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.data_enlarge:
    assert args.segment ==2
    order_prob = [0.1, 0.9]
else:
    order_prob = None
set_config("ic_parameter", {"segments": args.segment,
                            "batch_size": 128, "order_prob": order_prob})

VAR_KD = args.var_kd != "[1]"
USE_CF = args.trainaug .find("CF")>=0
USE_ADV = args.trainaug .find("ADV")>=0
USE_ENSEMBLE = args.ensemble != ""
HIST_AVG = args.hist_avg
DIST_WEIGHT = args.dist_weight != ""
ENSEMBLE_FAST = args.ensemble_fast
ENSEMBLE_BEST = args.ensemble_best
assert args.ensemble in ["snapshot", "bagging", ""]
assert args.dist_weight in ["I_normalized_l2", "I_normalized_l2_cap", "normalized_l2", ""]
DIST_WEIGHT_INV = args.dist_weight.find("I_")>=0
OCCULUTION = args.occulusion
DIST_WEIGHT_NORMALIZE = args.dist_weight.find("normalized")>=0
DIST_WEIGHT_CAP = args.dist_weight.find("cap")>=0
CON_IMPROVE = args.consistent_improve
MIX_LABEL = (args.mix_label != "")
MODEL_UPDATE = (args.model_update)
set_config("occulusion", OCCULUTION)
WARMUP = (args.warmup_ep != 0)

if MODEL_UPDATE:
    args.scratch = True
    args.net = "update"

# for one hot encoding
def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return T.mean(-(y_true * torch.log(y_pred)).sum(dim=1))

assert not args.unsup_kd or not args.correct_set
# Crop Flip
if args.trainaug == "":
    transform_train = transform_test
    Daug_method = ""
else:
    Daug_method = "#Aug_" + args.trainaug
if USE_CF :
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
else:
    transform_train = transform_test

if args.dataset == "cifar10":
    ds = torchvision.datasets.CIFAR10
elif args.dataset == "cifar100":
    ds = torchvision.datasets.CIFAR100
elif args.dataset == "imagenet32":
    ds = Imagenet32

if args.smalldata and args.unsup_kd:
    if args.unsupdata == "":
        sup_method = "#US_sameD"
    elif args.unsupdata == "imagenet":
        sup_method = "#US_ImageNet"
    else:
        assert False
else:
    sup_method = ""

if args.dataset in ["cifar10", "cifar100"]:
    proc_func = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    download = True
elif args.dataset in ["imagenet32"]:
    download = False
    proc_func = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
else:
    assert False

trainset = ds(
    root='./data', train=True, download=download)#, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=8)

testset = ds(
    root='./data', train=False, download=download)#, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=8)

#classes = ('plane', 'car', 'bird', 'cat', 'deer',
#           'dog', 'frog', 'horse', 'ship', 'truck')
print(type(trainset),type(testset),type(trainset+testset))
#torch.autograd.set_detect_anomaly(True)
# configure the max step

lwf = args.lwf
ewc = args.ewc
method_name = ""
if USE_ENSEMBLE:
    method_name += "#ensemble_{}_{}".format(args.ensemble_num,args.ensemble)
    granularity = "epoch"
    if ENSEMBLE_FAST:
        method_name += "_EFast"
    if ENSEMBLE_BEST:
        method_name += "_EBest"
    is_copy = False
else:
    granularity = "converge"
    is_copy = True
assert not DIST_WEIGHT or not args.seploss
if lwf:
    set_config("lwf_lambda", args.lwf_lambda)
    method_name += "#lwf{:.2e}".format(args.lwf_lambda)
    if args.unsup_kd:
        method_name += "_unsup"
    if args.correct_set:
        method_name += "#corrset"
    if args.seploss:
        method_name += "#seploss"   
if args.loss != "xent":
    method_name += "#loss{}".format(args.loss)    
if WARMUP:
    method_name += "Warmupep{}".format(args.warmup_ep) 
if ewc:
    set_config("ewc_lambda", args.ewc_lambda)
    method_name += "#ewc{:.2e}".format(args.ewc_lambda)
    #set_config("reset_head_before_task", True)
set_config("reset_net_before_task", args.scratch)
if DIST_WEIGHT:
    method_name += args.dist_weight

if VAR_KD:
    method_name += "#VKD_{}".format(str(args.var_kd))

if MIX_LABEL:
    method_name += "#mixL_{}".format(args.mix_label)
if args.scratch:
    method_name += "#scratch"
if HIST_AVG:
    method_name += "#hist_avg"
if args.interpolate:
    method_name += "#interpolate"
    set_config("ic_parameter",{"segments":3,"batch_size":128})
if CON_IMPROVE:
    method_name = "#improve_cp_step"

if args.kd_model != "":
    method_name += "#ext_kd_model"

if args.improve_loss:
    method_name += "#IMP_LOSSv4_Beta_{:.2}".format(args.improve_loss_beta)

if method_name == "":
    method_name = "#vanilla"

method_name += "_Opt{}".format(args.opt)
set_config("develop_assumption", args.dev_scene)
set_config("classification_task", args.inc_setting)
setting = incremental_config(args.dataset)
def init_model(order):
    global net, criterion, optimizer, ds_name
    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    torch.manual_seed(seed)
    netdct = {"ResNet18":ResNet18, "ResNet34":ResNet34, "ResNet152":ResNet152,"LeNet":CLeNet, "ResNet18v2":ResNet18}
    if MODEL_UPDATE:
        touse_models = ["LeNet", "ResNet18v2"]
        nname = touse_models[order]
        net =  netdct[nname](procfunc=proc_func)
    else:
        nname = args.net
        net =  netdct[args.net](procfunc=proc_func)
    os.makedirs("fix_init",exist_ok=True)
    cp_name = os.path.join("fix_init",ds_name + nname + str(seed))
    if os.path.exists(cp_name):
        net.load_state_dict(torch.load(cp_name))
    else:
        torch.save(net.state_dict(), cp_name)

    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    #net = SimpleDLA()
    net = net.to(device)
    if device == 'cuda':
        #net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterions = []
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    if args.loss.find("xent")>=0 :
        criterions.append(nn.CrossEntropyLoss(reduction="none")) 
    elif args.loss.find("l1")>=0:
        c = nn.L1Loss(reduction="none")
        def _criterion(output, label):
            return T.mean(c(F.softmax(output, dim=1), T.eye(get_config("CLASS_NUM"))[label].to(device)),dim=1)
        criterions.append(_criterion)
    def criterion(output, label):
        loss = 0
        for c in criterions:
            loss += c(output, label)
        return loss
    #optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                    momentum=0.9, weight_decay=5e-4)
    torch.manual_seed(seed)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)


if args.kd_model != "":
    external_model = ResNet18(procfunc=proc_func)
    path = "cifar10_small#Aug_CF_ResNet18_1.0e-02_#vanilla_DScifar10_CIMmask_CTdata_inc_DAsequential_CvgS40_DomS2_Ep200_FixInit_2_0.pth"
    checkpoint = torch.load(path)#args.kd_model)
    external_model.load_state_dict(checkpoint)


class ImageClassTraining(VT):

    def _model_process(self, task_name, model: nn.Module, key, step):
        if step == 0:
            global optimizer
            if get_config_default("reset_head_before_task", False):
                model.reset_head()
            if get_config("reset_net_before_task")  or USE_ENSEMBLE:
                init_model(self.curr_order_index)
                if USE_ENSEMBLE and not MODEL_UPDATE:
                    if ENSEMBLE_BEST and int(self.curr_order_index)>1:
                        net.load_state_dict(model.state_dict())
                    else:    
                        if self.curr_order_index!=0:
                            net.load_state_dict(model.estimators_[-1].state_dict())
                model = net

            torch.manual_seed(seed)
            if args.opt == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9, weight_decay=5e-4)
            elif args.opt == "adam":
                optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
            else:
                assert False

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
            if USE_ENSEMBLE:
                if args.ensemble == "snapshot":
                    model = SnapshotEnsembleClassifier(model, args.ensemble_num)
                elif args.ensemble == "bagging":
                    model = BaggingClassifier(model, args.ensemble_num)
                else:
                    assert False
                lr = args.lr
                weight_decay = 5e-4
                momentum = 0.9
                model.set_optimizer("SGD", lr=lr, weight_decay=weight_decay, momentum=momentum)
                if args.ensemble == "snapshot":
                    model.set_scheduler("CosineAnnealingLR", T_max=args.max_epoch/args.ensemble_num)
                elif args.ensemble == "bagging":
                    model.set_scheduler("CosineAnnealingLR", T_max=args.max_epoch)
        if step == -1:
            if ENSEMBLE_BEST and int(key)!=0:
                model = model.best_model
        return model

    def process_model4eval(self, modeldct):
        if args.interpolate and self.last_model is not None:
            for alpha in np.arange(0.1,1.0,step=0.1):
                avg_model = AvgNet()
                avg_model.add_net(modeldct[self.curr_task_name], weight = alpha)
                avg_model.add_net(self.last_model[self.curr_task_name], weight = 1 - alpha)
                self.evaluator.eval({self.curr_task_name:avg_model})
        if HIST_AVG:
            self.avg_model.add_net(modeldct[self.curr_task_name])
            return {self.curr_task_name:self.avg_model}
        else:
            return modeldct

    def pre_train(self):
        self.ewcs = {}
        if HIST_AVG:
            self.avg_model = AvgNet() 
        global ds_unsup
        if args.unsup_kd:
            self.dl_unsup = repeat_dataloader(self.process_data(ds_unsup, "eval"))
        if USE_ADV:
            self.fgs = FGS(epsilon=1.0/255,alpha = 1.0/4/255, min_val = 0.0, max_val = 1.0, max_iters = 8)
        if VAR_KD:
            var_kd = eval(args.var_kd)
            assert isinstance(var_kd, list)
            l = len(var_kd)
            self.epoch_to_kd_weight = []
            ep = args.max_epoch
            per_epoch = ep // l +1
            for i in range(ep):
                self.epoch_to_kd_weight.append(var_kd[i // per_epoch])
    def post_task(self):
        if ewc:
            _ewc = EWC(self.curr_train_data[self.curr_task_name], to_data_loader=partial(self.process_data, mode="eval"))
            _ewc.set_model(self.last_model[self.curr_task_name], self.task_var[self.curr_task_name])
            _ewc.eval_fisher()
            self.ewcs[self.curr_order] = _ewc
        
        torch.save(self.prev_models[self.curr_task_name][self.curr_order].state_dict(), "{}_{}_{}.pth".format(full_name,seed,self.curr_order))

    def cond_prob_guess(self, prev_prob, curr_prob, beta):
        max_prob = T.minimum(prev_prob,curr_prob)
        min_prob = prev_prob * curr_prob
        cond_prob_unnorm = (min_prob  * (1-beta) + max_prob * beta)/ (prev_prob+ 1e-5)        
        return cond_prob_unnorm

    def calculate_loss(self, oinputs, otargets, elemorder, model, compare_pairs, prev_models, metric, epoch):
        outputs_full = model(oinputs, full=True)
        outputs = model.process_output(outputs_full)
        targets = model.process_labels(otargets)

        
        loss_penalty = 0
        sample_weight = T.ones_like(targets)
        if len(compare_pairs) > 0:

            if (lwf or DIST_WEIGHT) and len(prev_models)>0:
                ### We use test mode to calculate for knowlege distillation loss
                #with PytorchModeWrap(model, False):
                #    outputs = model.process_output(outputs_full)
                #    targets = model.process_labels(otargets)                  
                for k in compare_pairs:
                    if args.kd_model != "":
                        kd_model = external_model
                    else:
                        kd_model = prev_models[k]
                    if USE_ENSEMBLE:
                        submodels = kd_model.estimators_
                        if ENSEMBLE_FAST:
                            l = len(submodels)
                            k1 = random.randrange(0,l)
                            submodels = [submodels[k1]]
                        
                    else:
                        submodels = [kd_model]
                    for kd_model in submodels:
                        with T.no_grad():
                            prev_full = kd_model(oinputs, full=True)
                            prev_output = kd_model.process_output(prev_full)
                        #bug? use full instead of processing output for metric
                        if lwf:
                            if args.correct_set:
                                mask = metric(prev_output, {"x":None,"y":otargets},kd_model)
                            else:
                                mask = None
                            if args.unsup_kd:
                                x_unsup, _ = next(self.dl_unsup)
                                x_kd = x_unsup.to(device)
                                kd_output = model(x_kd, full=True)
                                kd_prev_output = kd_model(x_kd, full=True)
                            else:
                                kd_output = outputs_full
                                kd_prev_output = prev_full
                            if args.seploss:
                                mask_prev = metric(prev_output, {"x":None,"y":otargets},kd_model)
                                mask_now = metric(outputs, {"x":None,"y":otargets},model)
                                mask = mask_prev * (1-mask_now)  # previously correct / now incorrect ones
                                sample_weight = (1- mask) 
                                sample_weight = sample_weight * sample_weight.size(0) / (T.sum(sample_weight) + 1e-2)
                            klg_loss = knowledge_distill_loss(kd_output, kd_prev_output, prev_models[k], mask=mask)
                            if VAR_KD:
                                klg_loss *= self.epoch_to_kd_weight[epoch]
                            loss_penalty += klg_loss / len(submodels)
                        if args.improve_loss:
                            Temp = 2.0
                            prev_prob = F.softmax(prev_output / Temp, dim=1)
                            curr_prob = F.softmax(outputs / Temp, dim=1)
                            """#version 2
                            max_prob = T.minimum(prev_prob,curr_prob)
                            min_prob = prev_prob * curr_prob
                            cond_prob_unnorm = (min_prob  * (1-args.improve_loss_beta) + max_prob * (args.improve_loss_beta))/ (prev_prob+ 1e-5)
                            cond_prob = cond_prob_unnorm #/ T.sum(cond_prob, dim=1,keepdim=True)
                            one_hot_prob = T.eye(get_config("CLASS_NUM"))[targets].to(device)
                            loss_penalty += categorical_cross_entropy(cond_prob,one_hot_prob)
                            """
                            """
                            #version 3
                            cond_prob_label = self.cond_prob_guess(prev_prob, curr_prob, args.improve_loss_beta)
                            #cond_prob_unlabel = 1.0 - self.cond_prob_guess(prev_prob, curr_prob, args.improve_loss_beta)
                            cond_prob_unlabel = self.cond_prob_guess(1.0 - prev_prob, 1.0 - curr_prob, args.improve_loss_beta)
                            one_hot_prob = T.eye(get_config("CLASS_NUM"))[targets].to(device)
                            final_cond = cond_prob_label * one_hot_prob + cond_prob_unlabel * (1.0-one_hot_prob)
                            loss_penalty += categorical_cross_entropy(final_cond,prev_prob)  
                            """
                            #Version 4
                            one_hot_prob = T.eye(get_config("CLASS_NUM"))[targets].to(device)
                            l_cond = T.less_equal(prev_prob,curr_prob).float()
                            g_cond = T.greater_equal(prev_prob,curr_prob).float()
                            mask = (1-one_hot_prob) * l_cond + one_hot_prob * g_cond
                            loss_penalty += categorical_cross_entropy(curr_prob,mask*prev_prob)  
                            
                        if DIST_WEIGHT:
                            _dist = T.sqrt(T.sum(T.square(prev_output - outputs), dim=1))
                            if DIST_WEIGHT_INV:
                                _dist = 1.0 / ( _dist + 1e-2)
                            if DIST_WEIGHT_NORMALIZE:
                                avg = T.mean(_dist)  
                                _dist = _dist / avg
                            if DIST_WEIGHT_CAP:
                                _dist = T.minimum(T.ones_like(_dist), _dist)
                            sample_weight = _dist.detach()

            if ewc and len(prev_models)>0:
                for k in compare_pairs:
                    loss_penalty += self.ewcs[k].penalty(model)#.module)
            loss_penalty /= len(compare_pairs)
        if MIX_LABEL and len(compare_pairs) > 0:
            if args.mix_label == "avg":
                with PytorchModeWrap(model, False):   
                    one_hot_prob = T.eye(get_config("CLASS_NUM"))[targets].to(device)               
                    for k in compare_pairs:
                        prev_full = prev_models[k](oinputs, full=True)
                        prev_output = prev_models[k].process_output(prev_full)
                        one_hot_prob += F.softmax(prev_output,dim=1)
                    one_hot_prob /= len(compare_pairs)+ 1
                    one_hot_prob = one_hot_prob.detach()
            
                loss = categorical_cross_entropy(F.softmax(outputs, dim=1),one_hot_prob)
            elif args.mix_label == "adpt_avg":
                one_hot_prob = T.eye(get_config("CLASS_NUM"))[targets].to(device)               

                one_hot_prob += F.softmax(outputs,dim=1)
                one_hot_prob /= 2
                one_hot_prob = one_hot_prob.detach()
                loss = categorical_cross_entropy(F.softmax(outputs, dim=1),one_hot_prob)
            else:
                assert False
        else:
            loss = criterion(outputs, targets)
        if WARMUP and len(compare_pairs) > 0:
            w = T.ones_like(targets).float()
            warmup_sample_weight = T.where(elemorder.eq(self.curr_order_index).to(device), w / args.warmup_ep * (epoch+1),  w )
            warmup_sample_weight = T.clamp(warmup_sample_weight,0.0, 1.0)
            sample_weight = warmup_sample_weight
        loss = T.mean(loss * sample_weight)
        loss += loss_penalty
        return loss, loss_penalty, outputs, targets

    def _train_single(self, omodel, dataloader, prev_models, device, epoch):
        print('\nEpoch: %d' % epoch)
        model = omodel#torch.nn.DataParallel(omodel)
        model.train()
        #model.module = omodel
        train_loss = 0
        correct = 0
        total = 0
        compare_pairs = [] 
                    
        for compare_pair in self.taskdata.comparison:   
           if compare_pair[-1] == self.curr_order:
                compare_pairs.append(compare_pair[0])        
        print("current compare pairs", compare_pairs)   
        metric = self.taskdata.get_metric(self.curr_task_name)  
        val = self.curr_val_data_loader[self.curr_task_name]
        if CON_IMPROVE:
            ## Make it larger if neccessary
            prev_corr = [None] * 100
            prev_state = copy.deepcopy(model.state_dict())
            prev_opt = copy.deepcopy(optimizer.state_dict())
        if len(compare_pairs)>0:
            prev_model = prev_models[compare_pairs[-1]]
        def improve():
            nonlocal model, val, prev_corr, prev_state, prev_opt
            if CON_IMPROVE and len(compare_pairs)>0:
                with torch.no_grad():
                    _con_sum = 0
                    _con_cnt = 0
                    with PytorchModeWrap(model, False):
                        f = True
                        for idx, (x, y) in enumerate(val):
                            #if idx>=5:
                            #    break
                            x, y = x.to(device), y.to(device)
                            if prev_corr[idx] is None:
                                output = prev_model(x)
                                mask = metric(output, {"x":None,"y":y},prev_model)
                                prev_corr[idx] = mask
                                _con_sum += 1
                                _con_cnt += 1
                            else:
                                output = model(x, full=True)
                                output =prev_model.process_output(output)
                                mask = metric(output, {"x":None,"y":y},model)                                
                                _con_sum += torch.sum( mask * prev_corr[idx])
                                #prev_corr[idx] = mask
                                _con_cnt += torch.sum(prev_corr[idx])
                        f = (_con_sum / _con_cnt) > 0.80
                    if f:
                        prev_state = copy.deepcopy(model.state_dict())
                        prev_opt = copy.deepcopy(optimizer.state_dict())
                        print("+",_con_sum / _con_cnt)
                    else:
                        print("-",_con_sum / _con_cnt)
                        model.load_state_dict(prev_state)
                        optimizer.load_state_dict(prev_opt)
        if USE_ENSEMBLE:
            val = self.curr_val_data_loader[self.curr_task_name]
            if args.ensemble == "snapshot":
                epochs = args.max_epoch
                assert isinstance(model, SnapshotEnsembleClassifier)
                model.fit(
                    dataloader,
                    epochs=epochs,
                    test_loader=val,
                    loss_func = lambda x, y, elemorder, model,epoch: self.calculate_loss(x, y, elemorder, model, compare_pairs, prev_models, metric, epoch)
                )
            elif args.ensemble == "bagging":
                epochs = args.max_epoch
                model.fit(
                    dataloader,
                    epochs=epochs,
                    test_loader=val,
                    loss_func = lambda x, y, elemorder, model,epoch: self.calculate_loss(x, y, elemorder, model, compare_pairs, prev_models, metric, epoch)
                )
            else:
                assert False
            self.evaluator.add_addition("uncertainty", evaluate_uncertainty(model, val).item())
            if len(compare_pairs)>0:
                for learnexpt in [True,False]:
                    for dataexpt in [True,False]:
                        self.evaluator.add_addition(f"uncertainty_{learnexpt}_{dataexpt}_{self.curr_order_index}", evaluate_uncertainty_part(prev_model ,model, dataexpt, learnexpt,  \
                            self.prev_train_data_loader[self.curr_task_name], val, self.prev_test_data_loader[self.curr_task_name]).item())
            if ENSEMBLE_BEST and len(compare_pairs)>0:
                best = -1
                best_model = None
                cs = []
                for cmodel in model.estimators_:
                    c = evaluate_consistency(prev_model, cmodel, val)
                    cs.append(c)
                    if c>best:
                        best = c
                        best_model = cmodel
                print(cs)
                model.best_model = best_model
        else:
            for batch_idx, (oinputs, otargets, elemorder) in enumerate(dataloader):
                #print(inputs.shape, targets.shape)
                improve()
                    #if batch_idx%10 == 0:
                        #print(idx, prev_corr[0][:1])
                        #print("mean", torch.mean(prev_corr[0].float()),torch.mean(mask) , f)
                oinputs, otargets = oinputs.to(device), otargets.to(device)
                if USE_ADV:
                    #print(oinputs.size())
                    oinputs = self.fgs.perturb(model, oinputs, model.process_labels(otargets),)
                
                optimizer.zero_grad()
                loss, loss_penalty, outputs, targets = self.calculate_loss(oinputs, otargets, elemorder, model, compare_pairs, prev_models, metric, epoch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            improve()
            self.scheduler.step()
            print(len(prev_models), loss_penalty)


    def _eval_single(self, model, dataloader, prev_models, device, epoch):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        print("eval")
        compare_pairs = []         
        for compare_pair in self.taskdata.comparison:   
           if compare_pair[-1] == self.curr_order:
                compare_pairs.append(compare_pair[0])        
        print("current compare pairs", compare_pairs)   
        metric = self.taskdata.get_metric(self.curr_task_name) 
        with torch.no_grad():
            for batch_idx, (oinputs, otargets) in enumerate(dataloader):
                oinputs, otargets = oinputs.to(device), otargets.to(device)
                
                loss, loss_penalty, outputs, targets = self.calculate_loss(oinputs, otargets, model, compare_pairs, prev_models, metric)
                _, predicted = outputs.max(1)
                test_loss += loss.item()*targets.size(0)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #            % (test_loss/total, 100.*correct/total, correct, total))
        print("eval loss", test_loss/total)
        return test_loss/total

    def process_data(self, dataset, mode, batch_size=None, sampler=None, shuffle=None):
        if batch_size is None:
            batch_size = 128
        assert mode in ["train", "eval", "test"]
        if shuffle is None:
            if mode == "train":
                shuffle = True
            else:
                shuffle = False
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, sampler=sampler, generator=torch.Generator().manual_seed(seed))

ICD = setting["taskdata"]
epsp = EPSP(device, max_step=  get_config("ic_parameter")["segments"])
seed = int(args.class_seed)


if args.smalldata:
    ds_name = args.dataset + "_small"
else:
    ds_name = args.dataset
full_name = "{}{}_{}_{:.1e}_{}{}_{}_Ep{}_FixInit".format(ds_name, Daug_method, args.net, args.lr, method_name, sup_method, get_config("full_name"), args.max_epoch)
path = os.path.join("./cl/results/", full_name, "Seed{}".format(seed))
if args.skip_exist:
    if  os.path.exists(path + "_config.json"):
        exit()
init_model(0)
# Get corresponding task data class based on different setting

#epsp.add_data(name="test",data=fd_test)
#epsp.add_data(name="train",data=fd_train)
IC_PARAM = get_config("ic_parameter")
print(cl.utils.config)
ds = ConcatDataset([trainset,testset])
if args.smalldata:
    if args.dataset in ["cifar10", "cifar100"]:
        ds, ds_remained = torch.utils.data.random_split(ds, [10000, 50000], generator=torch.Generator().manual_seed(seed))
    else:
        ds, ds_remained, _ = torch.utils.data.random_split(ds, [200000, 200000, len(ds)- 400000], generator=torch.Generator().manual_seed(seed))



if args.unsup_kd:
    if args.unsupdata == "":
        ds_unsup  = ds_remained
    elif args.unsupdata == "imagenet":
        from cl.dataset.d2lmdb import ImageFolderLMDB
        ds_unsup = ImageFolderLMDB("imagenet-train.ldmb")
    from cl.taskdata import OverideTransformDataset
    ds_unsup = OverideTransformDataset(ds_unsup, transform_train)
ic = ICD(ds, evaluator=epsp, metric =  MC(), segment_random_seed=seed, 
            training_transform=transform_train, 
            testing_transform=transform_test,
            **IC_PARAM)


train_cls = ImageClassTraining(max_epoch=args.max_epoch, granularity=granularity,\
        evalulator=epsp, taskdata=ic,task_prefix="cifar10_vanilla", iscopy =is_copy) #


train_cls.controlled_train_single_task(net)
os.makedirs(os.path.dirname(path), exist_ok=True)
epsp.save(path)
save_config(path)
#test(epoch)

