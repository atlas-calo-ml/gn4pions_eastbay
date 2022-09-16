#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pbatch
#BSUB -G hizphys
#BSUB -W 720
#BSUB -o outfiles/trackCalo_leading_deepset_20220613.out


source ~/.profile.coral

python train_trackCalo_leading.py --config gn4pions/configs/trackCalo_regress_deepset.yaml
