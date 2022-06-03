#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pbatch
#BSUB -G hizphys
#BSUB -W 720
#BSUB -o outfiles/trackCalo_all__20220602.out


source ~/.profile.coral

python train_trackCalo_all.py
