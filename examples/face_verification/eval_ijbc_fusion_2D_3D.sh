#!/bin/bash

# RESULT_DIR=results_ijbc_template
RESULT_DIR=results_ijbc_single_img


############################################################
############################################################
############################################################


# ResNet100 - Train: MS1MV3 -  1000 classes
DIST1=/home/bjgbiesseck/GitHub/InsightFace-tensorflow/results_ijbc_single_img/dataset=MS1MV3_1000subj_classes=1000_backbone=resnet-v2-m-50_epoch-num=100_margin=0.5_scale=64.0_lr=0.01_wd=0.0005_momentum=0.9_20230518-004011/ijbc.npy

# PointNeXt - Train: MS1MV3 -  1000 classes
# DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/results_ijbc_single_img/ms1mv3_3d_arcface-train-pointnext-s_arcface-ngpus1-seed6113-20230503-171328-SW2CTnmUDWBMoaVuSp4a4v/ijbc.npy

# PointNet++ - Train: MS1MV3 -  1000 classes
DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/results_ijbc_single_img/dataset=reconst_mica_ms1mv2_1000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=32_m=0.0_06052023_114705/ijbc.npy
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_fusion_2D_3D.py --result-dir $RESULT_DIR --dist1 $DIST1 --dist2 $DIST2

# ----------------

# ResNet100 - Train: MS1MV3 -  2000 classes
DIST1=/home/bjgbiesseck/GitHub/InsightFace-tensorflow/results_ijbc_single_img/dataset=MS1MV3_2000subj_classes=2000_backbone=resnet-v2-m-50_epoch-num=100_margin=0.5_scale=64.0_lr=0.01_wd=0.0005_momentum=0.9_20230518-010456/ijbc.npy

# PointNeXt - Train: MS1MV3 -  2000 classes
# DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/results_ijbc_single_img/ms1mv3_3d_arcface-train-pointnext-s_arcface-ngpus1-seed3095-20230601-183749-WJhGPMmF6JKjFykM4Uf7fg/ijbc.npy

# PointNet++ - Train: MS1MV3 -  2000 classes
DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/results_ijbc_single_img/dataset=reconst_mica_ms1mv2_2000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=32_m=0.0_09062023_184940/ijbc.npy
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_fusion_2D_3D.py --result-dir $RESULT_DIR --dist1 $DIST1 --dist2 $DIST2

# ----------------

# ResNet100 - Train: MS1MV3 -  5000 classes
DIST1=/home/bjgbiesseck/GitHub/InsightFace-tensorflow/results_ijbc_single_img/dataset=MS1MV3_classes=5000_backbone=resnet_v2_m_50_epoch-num=100_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.1_20230518-214716/ijbc.npy

# PointNeXt - Train: MS1MV3 -  5000 classes
# DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/results_ijbc_single_img/ms1mv3_3d_arcface-train-pointnext-s_arcface-ngpus1-seed2501-20230504-104157-7x3QoJexVmWZvcgzWuH2wV/ijbc.npy

# PointNet++ - Train: MS1MV3 -  5000 classes
DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/results_ijbc_single_img/dataset=reconst_mica_ms1mv2_5000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-06_moment=0.9_loss=arcface_s=32_m=0.0_12062023_154451/ijbc.npy
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_fusion_2D_3D.py --result-dir $RESULT_DIR --dist1 $DIST1 --dist2 $DIST2

# ----------------

# ResNet100 - Train: MS1MV3 - 10000 classes
DIST1=/home/bjgbiesseck/GitHub/InsightFace-tensorflow/results_ijbc_single_img/dataset=MS1MV3_classes=10000_backbone=resnet_v2_m_50_epoch-num=200_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.005_20230522-100202/ijbc.npy

# PointNeXt - Train: MS1MV3 - 10000 classes
# DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/results_ijbc_single_img/ms1mv3_3d_arcface-train-pointnext-s_arcface-ngpus1-seed345-20230511-151228-nQafgJFqP4i83ECK4VrzPg/ijbc.npy

# PointNet++ - Train: MS1MV3 - 10000 classes
DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/results_ijbc_single_img/dataset=reconst_mica_ms1mv2_10000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=32_m=0.0_26052023_222414/ijbc.npy
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_fusion_2D_3D.py --result-dir $RESULT_DIR --dist1 $DIST1 --dist2 $DIST2




############################################################
############################################################
############################################################




# ResNet100 - Train: WebFace260M -  1000 classes
DIST1=/home/bjgbiesseck/GitHub/InsightFace-tensorflow/results_ijbc_single_img/dataset=WebFace260M_1000subj_classes=1000_backbone=resnet_v2_m_50_epoch-num=100_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.01_20230524-142404/ijbc.npy

# PointNeXt - Train: WebFace260M -  1000 classes
# DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/results_ijbc_single_img/webface_3d_arcface-train-pointnext-s_arcface-ngpus1-seed3095-20230605-195440-iwUojCxmTg7o6SNF2JHpua/ijbc.npy

# PointNet++ - Train: WebFace260M -  1000 classes
DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/results_ijbc_single_img/dataset=reconst_mica_webface_1000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=16_m=0.0_05062023_194932/ijbc.npy
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_fusion_2D_3D.py --result-dir $RESULT_DIR --dist1 $DIST1 --dist2 $DIST2

# ----------------

# ResNet100 - Train: WebFace260M -  2000 classes
DIST1=/home/bjgbiesseck/GitHub/InsightFace-tensorflow/results_ijbc_single_img/dataset=WebFace260M_2000subj_classes=2000_backbone=resnet_v2_m_50_epoch-num=100_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.01_20230524-190517/ijbc.npy

# PointNeXt - Train: WebFace260M -  2000 classes
# DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/results_ijbc_single_img/webface_3d_arcface-train-pointnext-s_arcface-ngpus1-seed3095-20230605-232024-ChD5VgUfMqgp6r4SsQhrzL/ijbc.npy

# PointNet++ - Train: WebFace260M -  2000 classes
DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/results_ijbc_single_img/dataset=reconst_mica_webface_2000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=16_m=0.0_05062023_213735/ijbc.npy
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_fusion_2D_3D.py --result-dir $RESULT_DIR --dist1 $DIST1 --dist2 $DIST2

# ----------------

# ResNet100 - Train: WebFace260M -  5000 classes
DIST1=/home/bjgbiesseck/GitHub/InsightFace-tensorflow/results_ijbc_single_img/dataset=WebFace260M_5000subj_classes=5000_backbone=resnet_v2_m_50_epoch-num=150_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.01_20230525-093855/ijbc.npy

# PointNeXt - Train: WebFace260M -  5000 classes
# DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/results_ijbc_single_img/webface_3d_arcface-train-pointnext-s_arcface-ngpus1-seed3095-20230608-104723-F7xYY2Vnp4xS6EX28h3JxA/ijbc.npy

# PointNet++ - Train: WebFace260M -  5000 classes
DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/results_ijbc_single_img/dataset=reconst_mica_webface_5000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=32_m=0.0_06062023_235151/ijbc.npy
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_fusion_2D_3D.py --result-dir $RESULT_DIR --dist1 $DIST1 --dist2 $DIST2

# ----------------

# ResNet100 - Train: WebFace260M - 10000 classes
DIST1=/home/bjgbiesseck/GitHub/InsightFace-tensorflow/results_ijbc_single_img/dataset=WebFace260M_10000subj_classes=10000_backbone=resnet_v2_m_50_epoch-num=150_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.01_20230526-101421/ijbc.npy

# PointNeXt - Train: WebFace260M - 10000 classes
# DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/results_ijbc_single_img/webface_3d_arcface-train-pointnext-s_arcface-ngpus1-seed3095-20230614-093329-a5Gtk2dX6odcuCM23vy68Y/ijbc.npy

# PointNet++ - Train: WebFace260M - 10000 classes
DIST2=/home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/results_ijbc_single_img/dataset=reconst_mica_webface_10000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-06_moment=0.9_loss=arcface_s=32_m=0.0_13062023_123431/ijbc.npy
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_fusion_2D_3D.py --result-dir $RESULT_DIR --dist1 $DIST1 --dist2 $DIST2

