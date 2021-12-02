CUDA_VISIBLE_DEVICES=5 python eval_model.py --lr=0.1 --opt=sgd --small --dataset=cifar10 --inc-setting=data_inc --segment=2 --max-epoch=400 --ensemble=snapshot --model-path="cifar10_small#Aug_CF_ResNet18_1.0e-01_#ensemble_10_snapshot_Optsgd_DScifar10_CIMmask_CTdata_inc_DAsequential_CvgS40_DomS2_Ep200_FixInit_0_1.pth" --ensemble-num=10
uncertainty tensor(0.0618, device='cuda:0')
uncertainty tensor(0.0417, device='cuda:0')

CUDA_VISIBLE_DEVICES=5 python eval_model.py --lr=0.1 --opt=sgd --small --dataset=cifar10 --inc-setting=data_inc --segment=2 --max-epoch=400 --ensemble=snapshot --model-path="cifar10_small#Aug_CF_ResNet18_1.0e-01_#ensemble_5_snapshot_Optsgd_DScifar10_CIMmask_CTdata_inc_DAsequential_CvgS40_DomS2_Ep200_FixInit_0_1.pth" --ensemble-num=5
0.0424
uncertainty tensor(0.0342, device='cuda:0')

CUDA_VISIBLE_DEVICES=5 python eval_model.py --lr=0.1 --opt=sgd --small --dataset=cifar10 --inc-setting=data_inc --segment=2 --max-epoch=400 --ensemble=snapshot --model-path="cifar10_small#Aug_CF_ResNet18v2_1.0e-01_#ensemble_10_bagging_Optsgd_DScifar10_CIMmask_CTdata_inc_DAsequential_CvgS40_DomS2_OC_Ep100_FixInit_0_1.pth" --ensemble-num=10 --net="ResNet18v2"
uncertainty tensor(0.0701, device='cuda:0')
CUDA_VISIBLE_DEVICES=5 python eval_model.py --lr=0.1 --opt=sgd --small --dataset=cifar10 --inc-setting=data_inc --segment=2 --max-epoch=400 --ensemble=snapshot --model-path="cifar10_small#Aug_CF_ResNet18v2_1.0e-01_#ensemble_5_bagging_Optsgd_DScifar10_CIMmask_CTdata_inc_DAsequential_CvgS40_DomS2_OC_Ep100_FixInit_0_1.pth" --ensemble-num=5 --net="ResNet18v2"
uncertainty tensor(0.0611, device='cuda:0')