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
#args = parser.parse_args()

def run_cmd(device, trial, arg=None):
    if arg is None:
        args = " ".join(sys.argv[1:])
    else:
        args = arg
    cmd = 'CUDA_VISIBLE_DEVICES={} python ./main.py {} --class-seed={}'\
        .format(device, args, trial)
    print(cmd)
    os.system(cmd)

args = ["""--dataset="cifar10" --ewc --inc-setting="data_inc"  """,
        """--dataset="cifar10" --inc-setting="data_inc" """,
        """--dataset="cifar10" --lwf --inc-setting="domain_inc" """,
        """--dataset="cifar10" --inc-setting="domain_inc" """,
        """--dataset="cifar10" --ewc --inc-setting="domain_inc" """,
        """--dataset="cifar10" --lwf --inc-setting="data_inc" """,] 
Num = 4
results = []
p = Pool(Num)
for k in range(len(args)):
    for t in range(1):
        while True:
            try:        
                deviceID = GPUtil.getFirstAvailable(order = 'first', maxLoad=0.1, maxMemory=0.1, attempts=1, interval=60, verbose=False)[0]
                break
            except:
                pass
        args[k] += " --skip-exist"
        arg = (deviceID,t,args[k])
        kwarg = {}
        results.append( p.apply_async(run_cmd, arg, kwarg))
        time.sleep(30)
p.close()
p.join()
    