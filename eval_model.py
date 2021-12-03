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
from cl.algo.torchensemble import SnapshotEnsembleClassifier, evaluate_uncertainty, BaggingClassifier,  FastGeometricClassifier
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
parser.add_argument("--dist-weight",default="")
parser.add_argument("--unsup-kd",action="store_true")
parser.add_argument("--consistent-improve", action="store_true")
parser.add_argument("--net", default="ResNet18")
parser.add_argument("--lwf", action="store_true")
parser.add_argument("--loss",default="xent", choices=["l1","xent", "l1_xent"])
parser.add_argument("--lwf-lambda", default=1.0, type=float)
parser.add_argument("--scratch", action="store_true")
parser.add_argument("--ewc", action="store_true")
parser.add_argument("--ewc-lambda", default=5000.0)
parser.add_argument("--max-epoch", default=200,type=int)
parser.add_argument("--dev-scene", default="sequential")
parser.add_argument("--ensemble-num",default=5,type=int)
parser.add_argument("--inc-setting", default="data_inc")
parser.add_argument("--warmup-ep",default = 0, type=int)
parser.add_argument("--class-seed", default=0)
parser.add_argument("--kd-model", default = "")
parser.add_argument("--correct-set", action="store_true")
parser.add_argument("--var-kd",default="[1]")
parser.add_argument("--seploss", action="store_true")
parser.add_argument("--train-ext",action="store_true")
parser.add_argument("--opt",default="sgd")
parser.add_argument("--skip-exist",action="store_true")
parser.add_argument("--model-path",type=str,)
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

set_config("ic_parameter", {"segments": args.segment, "batch_size": 128})
VAR_KD = args.var_kd != "[1]"
USE_CF = args.trainaug .find("CF")>=0
USE_ADV = args.trainaug .find("ADV")>=0
USE_ENSEMBLE = args.ensemble != ""
HIST_AVG = args.hist_avg
DIST_WEIGHT = args.dist_weight != ""
assert args.ensemble in ["snapshot", "bagging", ""]
assert args.dist_weight in ["I_normalized_l2", "I_normalized_l2_cap", "normalized_l2", ""]
DIST_WEIGHT_INV = args.dist_weight.find("I_")>=0
DIST_WEIGHT_NORMALIZE = args.dist_weight.find("normalized")>=0
DIST_WEIGHT_CAP = args.dist_weight.find("cap")>=0
CON_IMPROVE = args.consistent_improve
MIX_LABEL = (args.mix_label != "")

WARMUP = (args.warmup_ep != 0)

# for one hot encoding
def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1)

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
    method_name += "#ensemble_5_{}".format(args.ensemble)
    granularity = "epoch"
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

if method_name == "":
    method_name = "#vanilla"

method_name += "_Opt{}".format(args.opt)
set_config("develop_assumption", args.dev_scene)
set_config("classification_task", args.inc_setting)
setting = incremental_config(args.dataset)
def init_model():
    global net, criterion, optimizer, ds_name
    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    torch.manual_seed(seed)
    netdct = {"ResNet18":ResNet18, "ResNet34":ResNet34, "ResNet152":ResNet152,"LeNet":CLeNet, "ResNet18v2":ResNet18}
    net =  netdct[args.net](procfunc=proc_func)
    os.makedirs("fix_init",exist_ok=True)
    cp_name = os.path.join("fix_init",ds_name + args.net+ str(seed))
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
            if get_config("reset_net_before_task") or USE_ENSEMBLE:
                init_model()
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
                model = SnapshotEnsembleClassifier(model, args.ensemble_num)
                lr = args.lr
                weight_decay = 5e-4
                momentum = 0.9
                model.set_optimizer("SGD", lr=lr, weight_decay=weight_decay, momentum=momentum)
                for i in range(args.ensemble_num):
                    model.estimators_.append(model._make_estimator())
        return model
    def _train_single(self, omodel, dataloader, prev_models, device, epoch):
        print('\nEpoch: %d' % epoch)
        model = omodel#torch.nn.DataParallel(omodel)
        model.eval()
        #model.module = omodel
        train_loss = 0
        correct = 0
        total = 0
        compare_pairs = [] 
            
        model.load_state_dict(torch.load(args.model_path))
        for compare_pair in self.taskdata.comparison:   
           if compare_pair[-1] == self.curr_order:
                compare_pairs.append(compare_pair[0])        
        print("current compare pairs", compare_pairs)   
        metric = self.taskdata.get_metric(self.curr_task_name)  
        val = self.curr_val_data_loader[self.curr_task_name]
        test = self.curr_test_data_loader[self.curr_task_name]

        if USE_ENSEMBLE:
            val = self.curr_val_data_loader[self.curr_task_name]
            if args.ensemble == "snapshot":
                epochs = args.max_epoch
                print("uncertainty on test", evaluate_uncertainty(model, test))
                print("uncertainty on val", evaluate_uncertainty(model, val))
            elif args.ensemble == "bagging":
                epochs = args.max_epoch
                print("uncertainty", evaluate_uncertainty(model, test))
                print("uncertainty on val", evaluate_uncertainty(model, val))
            else:
                assert False
        exit()

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
init_model()
# Get corresponding task data class based on different setting

#epsp.add_data(name="test",data=fd_test)
#epsp.add_data(name="train",data=fd_train)
IC_PARAM = get_config("ic_parameter")
print(cl.utils.config)
ds = ConcatDataset([trainset,testset])
if args.smalldata:
    if args.dataset in ["cifar10", "cifar100"]:
        ds, ds_remained = torch.utils.data.random_split(ds, [30000, 30000], generator=torch.Generator().manual_seed(seed))
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