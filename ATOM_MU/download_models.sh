#!/bin/bash

echo ""
echo "****************** ATOM Network ******************"
mkdir pytracking/networks
bash pytracking/utils/gdrive_download 1VNyr-Ds0khjM0zaq6lU-xfY74-iWxBvU pytracking/networks/atom_default.pth

echo "****************** Metric Network ******************"
bash pytracking/utils/gdrive_download 1o-btxlWWA6GlbwMGCGkzn2vAw9qv8D2z ../utils/metric_net/metric_model/metric_model.pt
echo ""
echo "****************** Download complete! ******************"

