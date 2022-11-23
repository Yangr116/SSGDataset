
WEIGHTS=/home/yr2/project/mmdetection_2_25_0/work_dirs/faster_rcnn_lb101_fold1/epoch_1.pth
# CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/faster_rcnn/faster_rcnn_lb101_fold1.py
# echo "*** ${CONFIG} ***"
# python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 benchmark.py $CONFIG $WEIGHTS --launcher 'pytorch'

# CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/cascade_rcnn/cascade_rcnn_lb101_fold1.py
# echo "*** ${CONFIG} ***"
# python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 benchmark.py $CONFIG $WEIGHTS --launcher 'pytorch'

# CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/retinanet/retinanet_lb101_fold1.py
# echo "*** ${CONFIG} ***"
# python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 benchmark.py $CONFIG $WEIGHTS --launcher 'pytorch'

# CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/fcos/fcos_lb101_fold1.py
# echo "*** ${CONFIG} ***"
# python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 benchmark.py $CONFIG $WEIGHTS --launcher 'pytorch'

# CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/atss/atss_r50_fpn_lb101_fold1.py
# echo "*** ${CONFIG} ***"
# python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 benchmark.py $CONFIG $WEIGHTS --launcher 'pytorch'

# CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/gfl/gfl_lb101_fold1.py
# echo "*** ${CONFIG} ***"
# python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 benchmark.py $CONFIG $WEIGHTS --launcher 'pytorch'

CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/swin/faster_rcnn_swin_t_lb101_fold1.py
echo "*** ${CONFIG} ***"
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 benchmark.py $CONFIG $WEIGHTS --launcher 'pytorch'

CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/pvt/faster_rcnn_pvt_s_lb101_fold1.py
echo "*** ${CONFIG} ***"
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 benchmark.py $CONFIG $WEIGHTS --launcher 'pytorch'

CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/scalablevit/faster_rcnn_scalablevit_s_lb101_fold1.py
echo "*** ${CONFIG} ***"
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 benchmark.py $CONFIG $WEIGHTS --launcher 'pytorch'

CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/uniformer/faster_rcnn_uniformer_lb101_fold1.py
echo "*** ${CONFIG} ***"
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 benchmark.py $CONFIG $WEIGHTS --launcher 'pytorch'
