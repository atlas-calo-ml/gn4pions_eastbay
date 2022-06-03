#!/bin/bash

# printf "\n\npython infer_tracks_noCells.py --save_dir results/onetrack_multicluster/Block_mae_20220516_1858_track_regress_noCell/\n"
# python infer_tracks_noCells.py --save_dir results/onetrack_multicluster/Block_mae_20220516_1858_track_regress_noCell/
# printf "\n\nDone!!"
# 
# printf "\n\npython infer_tracks_noCells.py --save_dir results/onetrack_multicluster/Block_mae_20220516_1859_track_regress_noCell_deepset/\n"
# python infer_tracks_noCells.py --save_dir results/onetrack_multicluster/Block_mae_20220516_1859_track_regress_noCell_deepset/
# printf "\n\nDone!!"

printf "\n\npython infer_tracks_noCells.py --save_dir results/onetrack_multicluster/Block_mae_20220516_1900_track_regress_noCell_nClusters/\n"
python infer_tracks_noCells.py --save_dir results/onetrack_multicluster/Block_mae_20220516_1900_track_regress_noCell_nClusters/
printf "\n\nDone!!"

printf "\n\npython infer_tracks_noCells.py --save_dir results/onetrack_multicluster/Block_mae_20220516_1901_track_regress_noCell_ncluster_deepset/\n"
python infer_tracks_noCells.py --save_dir results/onetrack_multicluster/Block_mae_20220516_1901_track_regress_noCell_ncluster_deepset/
printf "\n\nDone!!"
printf "\n\nAll Done!!"

