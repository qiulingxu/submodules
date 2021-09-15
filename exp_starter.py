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

for ds in ["imagenet32"]:
    for set in ["data_inc"]:#,"domain_inc"]:#"domain_inc", ]:
        for option1 in ["--lr=0.001 --smalldata"]:
            for method in ["--lwf", "--ewc", "--ensemble='snapshot'", "--hist-avg", "", "--scratch", "--lwf --correct-set","--lwf --unsup-kd", "--dist-weight='I_normalized_l2'", "--dist-weight='normalized_l2'"]:#
                for option2 in ["--trainaug='CF'"]:#, ""]: #
                    args.append(""" --dataset="{}" --inc-setting="{}" {}  {} {} """.
                        format(ds, set, method, option1, option2))
            for method in [""]:
                for option2 in [ "--trainaug='CF_ADV'"]: # "--trainaug='CF_ADV'",
                    args.append(""" --dataset="{}" --inc-setting="{}" {}  {} {} """.
                            format(ds, set, method, option1, option2))

for ds in ["imagenet32"]:
    for set in ["data_inc"]:#,"domain_inc"]:#"domain_inc", ]:
        for option1 in ["--lr=0.001 --smalldata"]:
            for method in ["--lwf --seploss", "--lwf --correct-set", "--lwf"]:#
                for lamda in [1.0, 0.5,  2.0, 0.1, 10.0]:
                    for option2 in ["--trainaug='CF'"]:#, ""]: #
                        args.append(""" --dataset="{}" --lwf-lambda={:.2f} --inc-setting="{}" {}  {} {} """.
                            format(ds, lamda, set, method, option1, option2))

"""for 5 classes cap need to be rerun July 29"""
for ds in ["cifar10"]:
    for set in ["data_inc"]:#,"domain_inc"]:#"domain_inc", ]:
        for option1 in ["--lr=0.001"]:
            for method in ["" ]:#
                for option2 in ["--dist-weight='I_normalized_l2_cap'", "--dist-weight='I_normalized_l2'", "--dist-weight='normalized_l2'"]: #
                        break
                        args.append(""" --dataset="{}" --inc-setting="{}" {}  {} {} """.
                            format(ds, set, method, option1, option2))

args = []
for ds in ["cifar10"]:
    for set in ["data_inc"]:#,"domain_inc"]:#"domain_inc", ]:
        for option1 in ["--lr=0.001 --small"]:
            for method in ["--lwf", "--ewc", "", "--scratch", "--loss=l1", "--loss=l1_xent", "--ensemble='snapshot'"]:#
                for option2 in [ ""]: #
                    break
                    args.append(""" --dataset="{}" --inc-setting="{}" {}  {} {} """.
                            format(ds, set, method, option1, option2))
            for method in ["--lwf"]:
                for st1 in [1,0.5,2,0.25,4]: #
                    for st2 in [1,0.5,2,0.25,4]: #
                        break
                        option2 = "--var-kd=\"{}\" --max-epoch=100".format(str([st1, st2]))
                        args.append(""" --dataset="{}" --inc-setting="{}" {}  {} {} """.
                                format(ds, set, method, option1, option2))
args = []
for ds in ["cifar10"]:#,"imagenet32"]:
    for set in ["data_inc"]:  # ,"domain_inc"]:#"domain_inc", ]:
        for option1 in ["--lr=0.1 --opt=sgd --small"]:
            for method in ["--ensemble='snapshot' "]:#"--lwf", "--lwf --ensemble='snapshot'", "--ensemble='snapshot'"]:
                for option2 in ["--segment=2", "--segment=4"]:
                    args.append(""" --dataset="{}" --inc-setting="{}" {}  {} {} """.
                                format(ds, set, method, option1, option2))

for ds in ["cifar10"]:
    for set in ["data_inc"]:#"domain_inc", ]:
        for method in ["--scratch --trainaug='CF' ",  "--ewc --trainaug='CF'", "--trainaug='CF_ADV'"]:#"--lwf --unsup-kd",
            for option in ["--lr=0.001 "]: #--smalldata
                break
                args.append(""" --dataset="{}" --inc-setting="{}" {}  {} """.
                    format(ds, set, method, option))
for ds in ["cifar10"]:
    for set in ["data_inc"]:#,"domain_inc"]:#"domain_inc", ]:
        for option1 in ["--lr=0.001"]:
            for method in ["--ensemble='snapshot'", "--hist-avg"]:#
                for option2 in ["--smalldata", ""]: #
                    break
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
                deviceID = GPUtil.getFirstAvailable(order = 'first', maxLoad=0.2, maxMemory=0.5, attempts=1, interval=60, verbose=False)[0]
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
    
