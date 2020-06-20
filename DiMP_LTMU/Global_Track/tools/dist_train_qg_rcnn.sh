# baseline
python -m torch.distributed.launch --nproc_per_node=4 tools/train_qg_rcnn.py \
    --launcher pytorch --load_from checkpoints/qg_rcnn_r50_fpn_2x_20181010-443129e1.pth \
    --base_dataset "coco_train" --sampling_prob "1" \
    --gpus 4 --work_dir work_dirs/qg_rcnn_r50_fpn_baseline

# COCO+GOT+LaSOT
python -m torch.distributed.launch --nproc_per_node=4 tools/train_qg_rcnn.py \
    --launcher pytorch --load_from work_dirs/qg_rcnn_r50_fpn_baseline/epoch_12.pth \
    --base_dataset "coco_train,got10k_train,lasot_train" --sampling_prob "0.4,0.4,0.2" \
    --gpus 4 --work_dir work_dirs/qg_rcnn_r50_fpn_coco_got10k_lasot
