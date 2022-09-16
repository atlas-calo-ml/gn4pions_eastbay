#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pbatch
#BSUB -G hizphys
#BSUB -W 720
#BSUB -o outfiles/trackCalo_all_deepsets_20220711.out


source ~/.profile.coral

python train_trackCalo_all.py --config gn4pions/configs/trackMultiCalo_regress_deepset.yaml
