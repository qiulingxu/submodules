'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

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
from cl.utils import get_config, save_config, set_config
from cl.algo import knowledge_distill_loss, EWC
from functools import partial
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--dataset", default="cifar10")
parser.add_argument("--net", default="ResNet18")
parser.add_argument("--lwf", action="store_true")
parser.add_argument("--ewc", action="store_true")
parser.add_argument("--dev-scene", default="sequential")
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


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=8)

#classes = ('plane', 'car', 'bird', 'cat', 'deer',
#           'dog', 'frog', 'horse', 'ship', 'truck')

# configure the max step
lwf = args.lwf
ewc = args.ewc
set_config("develop_assumption", args.dev_scene)
set_config("classification_task", "domain_inc")
config = incremental_config(args.dataset)

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
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)


class ImageClassTraining(VT):

    def _model_process(self, model: nn.Module, key, step):
        if step == 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    def pre_train(self):
        self.ewcs = {}

            

    def post_task(self):
        if ewc:
            _ewc = EWC(self.curr_train_data, to_data_loader=partial(self.process_data, mode="test"))
            _ewc.set_model(self.curr_model)
            _ewc.eval()
            self.ewcs[self.curr_order] = _ewc


    def _train(self, omodel, dataloader, prev_models, device, epoch):
        print('\nEpoch: %d' % epoch)
        model = torch.nn.DataParallel(omodel)
        train_loss = 0
        correct = 0
        total = 0
        compare_pairs = []         
        for compare_pair in self.taskdata.comparison:   
            if compare_pair[-1] == self.current_k:
                compare_pairs.append(compare_pair[0])             
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            #print(inputs.shape, targets.shape)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_penalty = 0
            for k in compare_pairs:
                if lwf and len(prev_models)>0:
                    klg_loss = knowledge_distill_loss(model, prev_models[k],)
                    loss_penalty += klg_loss
                if ewc and len(prev_models)>0:
                    loss_penalty += self.ewcs[k].penalty(model)
            loss_penalty /= len(compare_pairs)
            loss += loss_penalty
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        self.scheduler.step()

    def _eval(self, model, dataloader, prev_models, device, epoch):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        print("eval")
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def process_data(self, dataset, mode, batch_size=None):
        if batch_size is None:
            batch_size = 128
        return torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True, num_workers=2)


ICD = config["taskdata"]
epsp = EPSP(MC(), device, max_step= 6)
#epsp.add_data(name="test",data=fd_test)
#epsp.add_data(name="train",data=fd_train)
IC_PARAM = get_config("ic_parameter")
ic = ICD(trainset+testset, IC_PARAM, evaluator=epsp)

train_cls = ImageClassTraining(max_epoch=100, granularity="converge",\
        evalulator=epsp, taskdata=ic,task_prefix="cifar10_vanilla")


train_cls.controlled_train(net)
epsp.save("./cl/results/cifar10_5_domain_inc")
save_config("./cl/results/cifar10_5_domain_inc")
#test(epoch)

