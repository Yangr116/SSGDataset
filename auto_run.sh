
# CONFIG=./configs_defect_screen/faster_rcnn/faster_rcnn_full.py
# bash ./tools/dist_train.sh $CONFIG 4

# CONFIG=./configs_defect_screen/faster_rcnn/faster_rcnn_full_v2.py
# bash ./tools/dist_train.sh $CONFIG 4


# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/retinanet_pvt_s_lb101_fold1.py 8

# CONFIG=./configs_defect_screen/faster_rcnn/faster_rcnn_lb101_fold2.py
# CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash ./tools/dist_train.sh $CONFIG 4

# bash ./tools/dist_test.sh $CONFIG /home/yr2/project/mmdetection_2_25_0/work_dirs/faster_rcnn_full_v2/epoch_12.pth 4 --eval bbox

# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/retinanet_pvt_s_lb101_fold1.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/retinanet_pvt_s_lb101_fold2.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/retinanet_pvt_s_lb101_fold3.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/retinanet_pvt_s_lb101_fold4.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/retinanet_pvt_s_lb101_fold5.py 8

# faster swin-t
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/swin/faster_rcnn_swin_t_lb101_fold1.py 8
PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/swin/faster_rcnn_swin_t_lb101_fold2.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/swin/faster_rcnn_swin_t_lb101_fold3.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/swin/faster_rcnn_swin_t_lb101_fold4.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/swin/faster_rcnn_swin_t_lb101_fold5.py 8

# # faster swin-t 201
# killall python3; killall python
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/swin/faster_rcnn_swin_t_lb201_fold1.py 8
PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/swin/faster_rcnn_swin_t_lb201_fold2.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/swin/faster_rcnn_swin_t_lb201_fold3.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/swin/faster_rcnn_swin_t_lb201_fold4.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/swin/faster_rcnn_swin_t_lb201_fold5.py 8

# # faster pvt-s
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/faster_rcnn_pvt_s_lb101_fold1.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/faster_rcnn_pvt_s_lb101_fold2.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/faster_rcnn_pvt_s_lb101_fold3.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/faster_rcnn_pvt_s_lb101_fold4.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/faster_rcnn_pvt_s_lb101_fold5.py 8

# # pvt 201
# killall python3; killall python
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/faster_rcnn_pvt_s_lb201_fold1.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/faster_rcnn_pvt_s_lb201_fold2.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/faster_rcnn_pvt_s_lb201_fold3.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/faster_rcnn_pvt_s_lb201_fold4.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/pvt/faster_rcnn_pvt_s_lb201_fold5.py 8

# # uniformer 101
# killall python3; killall python
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/uniformer/faster_rcnn_uniformer_lb101_fold1.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/uniformer/faster_rcnn_uniformer_lb101_fold2.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/uniformer/faster_rcnn_uniformer_lb101_fold3.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/uniformer/faster_rcnn_uniformer_lb101_fold4.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/uniformer/faster_rcnn_uniformer_lb101_fold5.py 8

# killall python3; killall python
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/uniformer/faster_rcnn_uniformer_lb201_fold1.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/uniformer/faster_rcnn_uniformer_lb201_fold2.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/uniformer/faster_rcnn_uniformer_lb201_fold3.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/uniformer/faster_rcnn_uniformer_lb201_fold4.py 8
# PORT=29500 bash ./tools/dist_train.sh ./configs_defect_screen/uniformer/faster_rcnn_uniformer_lb201_fold5.py 8

# killall python3; killall python
# PORT=29500 bash ./tools/dist_train.sh  ./configs_defect_screen/scalablevit/faster_rcnn_scalablevit_s_lb101_fold1.py 8
# PORT=29500 bash ./tools/dist_train.sh  ./configs_defect_screen/scalablevit/faster_rcnn_scalablevit_s_lb101_fold2.py 8
# PORT=29500 bash ./tools/dist_train.sh  ./configs_defect_screen/scalablevit/faster_rcnn_scalablevit_s_lb101_fold3.py 8
# PORT=29500 bash ./tools/dist_train.sh  ./configs_defect_screen/scalablevit/faster_rcnn_scalablevit_s_lb101_fold4.py 8
# PORT=29500 bash ./tools/dist_train.sh  ./configs_defect_screen/scalablevit/faster_rcnn_scalablevit_s_lb101_fold5.py 8

# killall python3; killall python
# PORT=29500 bash ./tools/dist_train.sh  ./configs_defect_screen/scalablevit/faster_rcnn_scalablevit_s_lb201_fold1.py 8 # 重跑
# PORT=29500 bash ./tools/dist_train.sh  ./configs_defect_screen/scalablevit/faster_rcnn_scalablevit_s_lb201_fold2.py 8
# PORT=29500 bash ./tools/dist_train.sh  ./configs_defect_screen/scalablevit/faster_rcnn_scalablevit_s_lb201_fold3.py 8
# PORT=29500 bash ./tools/dist_train.sh  ./configs_defect_screen/scalablevit/faster_rcnn_scalablevit_s_lb201_fold4.py 8
# PORT=29500 bash ./tools/dist_train.sh  ./configs_defect_screen/scalablevit/faster_rcnn_scalablevit_s_lb201_fold5.py 8
# PORT=29500 bash ./tools/dist_train.sh  ./configs_defect_screen/scalablevit/faster_rcnn_scalablevit_s_lb201_fold1.py 8
