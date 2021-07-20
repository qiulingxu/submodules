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
    cmd = 'CUDA_VISIBLE_DEVICES={} python ./main.py {} --class-seed={}'\
        .format(device, args, trial)
    print(cmd)
    os.system(cmd)

args = [#"""--dataset="cifar10" --ewc --inc-setting="data_inc"  """,
        """--dataset="cifar10" --lwf --inc-setting="data_inc" """,
        #"""--dataset="cifar10" --inc-setting="data_inc" """,
        """--dataset="cifar10" --lwf --inc-setting="domain_inc" """,]
        #"""--dataset="cifar10" --inc-setting="domain_inc" """,
        #"""--dataset="cifar10" --ewc --inc-setting="domain_inc" """,]
        #] 
args = []
for ds in ["cifar10"]:
    for set in ["data_inc"]:#,"domain_inc"]:#"domain_inc", ]:
        for option1 in ["--lr=0.001 --smalldata"]:
            for method in ["--lwf", "--ewc", "", "--scratch", "--lwf --correct-set","--lwf --unsup-kd", ]:#
                for option2 in ["--trainaug='CF'", ""]: #
                        args.append(""" --dataset="{}" --inc-setting="{}" {}  {} {} """.
                            format(ds, set, method, option1, option2))
            for method in [""]:
                for option2 in ["--trainaug='CF_ADV'", "--trainaug='ADV'"]: #
                        args.append(""" --dataset="{}" --inc-setting="{}" {}  {} {} """.
                            format(ds, set, method, option1, option2))

for ds in ["cifar10"]:
    for set in ["data_inc"]:#,"domain_inc"]:#"domain_inc", ]:
        for option1 in ["--lr=0.001"]:
            for method in ["--lwf", "--ewc", "", "--scratch", "--lwf --correct-set" ]:#
                for option2 in ["--trainaug='CF'", ""]: #
                        args.append(""" --dataset="{}" --inc-setting="{}" {}  {} {} """.
                            format(ds, set, method, option1, option2))
            for method in [""]:
                for option2 in ["--trainaug='CF_ADV'", "--trainaug='ADV'"]: #
                        args.append(""" --dataset="{}" --inc-setting="{}" {}  {} {} """.
                            format(ds, set, method, option1, option2))


for ds in ["cifar10"]:
    for set in ["data_inc"]:#"domain_inc", ]:
        for method in ["--scratch --trainaug='CF' ",  "--ewc --trainaug='CF'", "--trainaug='CF_ADV'"]:#"--lwf --unsup-kd",
            for option in ["--lr=0.001 "]: #--smalldata
                break
                args.append(""" --dataset="{}" --inc-setting="{}" {}  {} """.
                    format(ds, set, method, option))
Num = 8
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
        args[k] += ""#--smalldata" # --skip-exist 
        arg = (deviceID,t,args[k])
        kwarg = {}
        results.append( p.apply_async(run_cmd, arg, kwarg))
        time.sleep(30)
p.close()
p.join()
    