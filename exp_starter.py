import argparse
import sys
import GPUtil
from multiprocessing import Pool # Pool
import os
import subprocess
import time
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--dataset", default="cifar10")
parser.add_argument("--net", default="ResNet18")
parser.add_argument("--lwf", action="store_true")
parser.add_argument("--lwf-lambda", default=1.0)
parser.add_argument("--ewc", action="store_true")
parser.add_argument("--ewc-lambda", default=1.0)
parser.add_argument("--dev-scene", default="sequential")
parser.add_argument("--class-seed")
args = parser.parse_args()

def run_cmd(device, trial):
    
    args = " ".join(sys.argv[1:])
    cmd = 'CUDA_VISIBLE_DEVICES={} python ./main.py {} --class-seed={}'\
        .format(device, args, trial)
    print(cmd)
    os.system(cmd)

Num = 4
results = []
p = Pool(Num)
for t in range(5):
    deviceID = GPUtil.getFirstAvailable(order = 'first', maxLoad=0.1, maxMemory=0.1, attempts=1, interval=60, verbose=False)[0]
    arg = (deviceID,t)
    kwarg = {}
    results.append( p.apply_async(run_cmd, arg, kwarg))
    time.sleep(30)
p.close()
p.join()
    