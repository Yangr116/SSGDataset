# CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/faster_rcnn/faster_rcnn_lb101_fold1.py
# echo "*** ${CONFIG} ***"
# python3 get_flops.py $CONFIG

# CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/cascade_rcnn/cascade_rcnn_lb101_fold1.py
# echo "*** ${CONFIG} ***"
# python3 get_flops.py $CONFIG  

# CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/retinanet/retinanet_lb101_fold1.py
# echo "*** ${CONFIG} ***"
# python3 get_flops.py $CONFIG  

# CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/fcos/fcos_lb101_fold1.py
# echo "*** ${CONFIG} ***"
# python3 get_flops.py $CONFIG  

# CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/atss/atss_r50_fpn_lb101_fold1.py
# echo "*** ${CONFIG} ***"
# python3 get_flops.py $CONFIG  

# CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/gfl/gfl_lb101_fold1.py

CONFIG=/home/yr2/project/mmdetection_2_25_0/configs_defect_screen/uniformer/faster_rcnn_uniformer_lb101_fold1.py
echo "*** ${CONFIG} ***"
python3 get_flops.py $CONFIG
