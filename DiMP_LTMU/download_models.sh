#!/bin/bash

echo ""
echo "****************** DiMP Network ******************"
mkdir pytracking/networks
bash pytracking/utils/gdrive_download 1qgachgqks2UGjKx-GdO1qylBDdB1f9KN pytracking/networks/dimp50.pth

echo "****************** Siammask Network ******************"

cd SiamMask/experiments/siammask
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT_LD.pth
cd ../../../

echo "****************** Global_Track Network ******************"
mkdir Global_Track/checkpoints
bash pytracking/utils/gdrive_download 1ZTdQbZ1tyN27UIwUnUrjHChQb5ug2sxr Global_Track/checkpoints/qg_rcnn_r50_fpn_coco_got10k_lasot.pth

echo "****************** Metric Network ******************"
bash pytracking/utils/gdrive_download 1o-btxlWWA6GlbwMGCGkzn2vAw9qv8D2z ../utils/metric_net/metric_model/metric_model.pt
echo ""
echo "****************** Download complete! ******************"

