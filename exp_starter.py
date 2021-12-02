import argparse
import sys
import GPUtil
from multiprocessing import Pool # Pool
import os
import subprocess
import time

#args = parser.parse_args()

def run_cmd(device, trial, arg=None):
    if arg is None:
        args = " ".join(sys.argv[1:])
    else:
        args = arg
    cmd = 'CUDA_VISIBLE_DEVICES={} python ./main.py {} --class-seed={} '\
        .format(device, args, trial) #--skip-exist
    print(cmd)
    os.system(cmd)

args = []


args = []
for ds in ["cifar10", "imagenet32"]:#,"imagenet32"]:
    for set in ["data_inc"]:  # ,"domain_inc"]:#"domain_inc", ]:
        for option1 in ["--lr=0.1 --opt=sgd --max-epoch=100 --segment=2"]:
            for option2 in ["--model-update", "--occulusion", "--data-enlarge"]:  #
                for method in ["", "--lwf","--lwf --ensemble='snapshot'",  
                        "--lwf --ensemble='bagging'",
                        ]:#["--ensemble-num=5 --lwf --ensemble='bagging' ", "--ensemble='bagging' ","--lwf", ]:#["--ensemble='snapshot' ", "", "--lwf", "--lwf --ensemble='snapshot'",]:
                    args.append(""" --dataset="{}" --inc-setting="{}" {}  {} {} """.
                                format(ds, set, method, option1, option2))

# python eval_model.py --lr=0.1 --ensemble='snapshot' --opt=sgd --small --dataset="cifar10" --inc-setting="data_inc"  --model-path = "cifar10_small#Aug_CF_ResNet18_1.0e-01_#ensemble_5_snapshot_Optsgd_DScifar10_CIMmask_CTdata_inc_DAsequential_CvgS40_DomS2_Ep200_FixInit_0_0.pth"                  
Num = 8
results = []
p = Pool(Num)
for k in range(len(args)):
    for t in range(1):
        while True:
            try:        
                deviceID = GPUtil.getFirstAvailable(
                    order='first', maxLoad=0.2,  maxMemory=0.5, attempts=1, interval=60, verbose=False)[0]  # excludeID =[0,1,2,3,4],
                break
            except:
                pass
        args[k] += ""#--smalldata" # --skip-exist 
        arg = (deviceID,t,args[k])
        kwarg = {}
        results.append( p.apply_async(run_cmd, arg, kwarg))
        time.sleep(120)
p.close()
p.join()
    
#CUDA_VISIBLE_DEVICES = {GPUID} python ./main.py --model-update --class-seed=0  --lr=0.1 --opt=sgd --max-epoch=100 --segment=2 --dataset="cifar10" --inc-setting="data_inc"
