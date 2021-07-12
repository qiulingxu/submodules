'''Train CIFAR10 with PyTorch.'''
from numpy import full
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import optimizer

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from cl import EvalProgressPerSampleClassification as EPSP, \
    FixDataMemoryBatchClassification as FD, \
    MetricClassification as MC, \
    ClassificationTrain as VT, \
    ClassificationMask as CM
from cl.configs.imageclass_config import incremental_config
from cl.utils import get_config, get_config_default, save_config, set_config, repeat_dataloader
from cl.algo import knowledge_distill_loss, EWC
from torch.utils.data import ConcatDataset
import cl
from functools import partial
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--dataset", default="cifar10")
parser.add_argument("--smalldata",action="store_true")
parser.add_argument("--unsupdata",default="")
parser.add_argument("--unsup-kd",action="store_true")
parser.add_argument("--net", default="ResNet18")
parser.add_argument("--lwf", action="store_true")
parser.add_argument("--lwf-lambda", default=1.0)
parser.add_argument("--scratch", action="store_true")
parser.add_argument("--ewc", action="store_true")
parser.add_argument("--ewc-lambda", default=5000.0)
parser.add_argument("--dev-scene", default="sequential")
parser.add_argument("--inc-setting", default="domain_inc")
parser.add_argument("--class-seed", default=0)
parser.add_argument("--correct-set", action="store_true")
parser.add_argument("--skip-exist",action="store_true")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == "cifar10":
    ds = torchvision.datasets.CIFAR10
elif args.dataset == "cifar100":
    ds = torchvision.datasets.CIFAR100

if args.smalldata and args.unsup_kd:
    if args.unsupdata == "":
        sup_method = "#US_sameD"
    elif args.unsupdata == "imagenet":
        sup_method = "#US_ImageNet"
    else:
        assert False
else:
    sup_method = ""

trainset = ds(
    root='./data', train=True, download=True)#, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=8)

testset = ds(
    root='./data', train=False, download=True)#, transform=transform_test)
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

if lwf:
    set_config("lwf_lambda", args.lwf_lambda)
    method_name += "#lwf{:.2e}".format(args.lwf_lambda)
    if args.unsup_kd:
        method_name += "_unsup"
    if args.correct_set:
        method_name += "#corrset"
if ewc:
    set_config("ewc_lambda", args.ewc_lambda)
    method_name += "#ewc{:.2e}".format(args.ewc_lambda)
    #set_config("reset_head_before_task", True)
set_config("reset_net_before_task", args.scratch)
if args.scratch:
    method_name += "#scratch"

if method_name == "":
    method_name = "#vanilla"
set_config("develop_assumption", args.dev_scene)
set_config("classification_task", args.inc_setting)
setting = incremental_config(args.dataset)
def init_model():
    global net, criterion, optimizer
    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    net =  ResNet18()
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

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                    momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)


class ImageClassTraining(VT):

    def _model_process(self, task_name, model: nn.Module, key, step):
        if step == 0:
            if get_config_default("reset_head_before_task", False):
                model.reset_head()
            if get_config("reset_net_before_task"):
                init_model()
                model = net
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return model
    def pre_train(self):
        self.ewcs = {}
        if args.unsup_kd:
            self.dl_unsup = repeat_dataloader(self.process_data(ds_unsup, "test"))

            

    def post_task(self):
        if ewc:
            _ewc = EWC(self.curr_train_data[self.curr_task_name], to_data_loader=partial(self.process_data, mode="test"))
            _ewc.set_model(self.last_model[self.curr_task_name], self.task_var[self.curr_task_name])
            _ewc.eval_fisher()
            self.ewcs[self.curr_order] = _ewc

    def calculate_loss(self, oinputs, otargets, model, compare_pairs, prev_models, metric):
        outputs_full = model(oinputs, full=True)
        outputs = model.process_output(outputs_full)
        targets = model.process_labels(otargets)
        loss = criterion(outputs, targets)
        loss_penalty = 0
        if len(compare_pairs) > 0:
            for k in compare_pairs:
                if lwf and len(prev_models)>0:
                    prev_full = prev_models[k](oinputs, full=True)
                    prev_output = prev_models[k].process_output(prev_full)
                    #bug? use full instead of processing output for metric
                    if args.correct_set:
                        mask = metric(prev_output, {"x":None,"y":otargets},prev_models[k])
                    else:
                        mask = None
                    if args.unsup_kd:
                        x_unsup, _ = next(self.dl_unsup)
                        x_kd = x_unsup.to(device)
                    else:
                        x_kd = oinputs
                    klg_loss = knowledge_distill_loss(outputs_full, prev_full, prev_models[k], x_kd, mask=mask)
                    loss_penalty += klg_loss
                if ewc and len(prev_models)>0:
                    loss_penalty += self.ewcs[k].penalty(model)#.module)
            loss_penalty /= len(compare_pairs)
        loss += loss_penalty
        return loss, loss_penalty, outputs, targets

    def _train_single(self, omodel, dataloader, prev_models, device, epoch):
        print('\nEpoch: %d' % epoch)
        model = omodel#torch.nn.DataParallel(omodel)
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
        for batch_idx, (oinputs, otargets) in enumerate(dataloader):
            #print(inputs.shape, targets.shape)
            oinputs, otargets = oinputs.to(device), otargets.to(device)
            optimizer.zero_grad()
            loss, loss_penalty, outputs, targets = self.calculate_loss(oinputs, otargets, model, compare_pairs, prev_models, metric)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
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

    def process_data(self, dataset, mode, batch_size=None, sampler=None, shuffle=True):
        if batch_size is None:
            batch_size = 128
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, sampler=sampler)

ICD = setting["taskdata"]
epsp = EPSP(device, max_step= 25)
seed = int(args.class_seed)

if args.smalldata:
    ds_name = args.dataset + "_small"
else:
    ds_name = args.dataset
full_name = "{}_{}_{:.1e}_{}{}_{}".format(ds_name, args.net, args.lr, method_name, sup_method, get_config("full_name"))
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
    ds, ds_remained = torch.utils.data.random_split(ds, [10000, 50000], generator=torch.Generator().manual_seed(seed))
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


train_cls = ImageClassTraining(max_epoch=100, granularity="converge",\
        evalulator=epsp, taskdata=ic,task_prefix="cifar10_vanilla") #


train_cls.controlled_train_single_task(net)
os.makedirs(os.path.dirname(path), exist_ok=True)
epsp.save(path)
save_config(path)
#test(epoch)

