#!/usr/bin/env sh
now=$(date +"%Y%m%d_%H%M%S")
GLOG_logtostderr=1 MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 LD_LIBRARY_PATH=/mnt/lustre/share/opencv/lib:/mnt/lustre/share/cuda-8.0/lib64 srun --mpi=pmi2 --gres=gpu:4 --ntasks=1 --partition=Face python -u  experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/resnet_v1_101_imagenet_vid_rfcn_end2end_ohem.yaml 2>&1|tee log/train-$now.log
