#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pbatch
#BSUB -G hizphys
#BSUB -W 720
#BSUB -o outfiles/tracks_noCells_20220613.out

source ~/.profile.coral

python train_tracks_noCells.py --config gn4pions/configs/track_regress_noCell.yaml &
sleep 1m

python train_tracks_noCells.py --config gn4pions/configs/track_regress_noCell_deepset.yaml &
sleep 1m

python train_tracks_noCells.py --config gn4pions/configs/track_regress_noCell_nClusters.yaml &
sleep 1m

python train_tracks_noCells.py --config gn4pions/configs/track_regress_noCell_ncluster_deepset.yaml

