#!/bin/bash

# PointNeXt - Train: MS1MV3 -  1000 classes
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_single_image.py --num_points 1024 --config-path /home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/log/ms1mv3_3d_arcface/ms1mv3_3d_arcface-train-pointnext-s_arcface-ngpus1-seed6113-20230503-171328-SW2CTnmUDWBMoaVuSp4a4v/pointnext-s_arcface.yaml

# PointNeXt - Train: MS1MV3 -  2000 classes
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_single_image.py --num_points 1024 --config-path /home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/log/ms1mv3_3d_arcface/ms1mv3_3d_arcface-train-pointnext-s_arcface-ngpus1-seed3095-20230601-183749-WJhGPMmF6JKjFykM4Uf7fg/pointnext-s_arcface.yaml

# PointNeXt - Train: MS1MV3 -  5000 classes
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_single_image.py --num_points 2048 --config-path /home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/log/ms1mv3_3d_arcface/ms1mv3_3d_arcface-train-pointnext-s_arcface-ngpus1-seed2501-20230504-104157-7x3QoJexVmWZvcgzWuH2wV/pointnext-s_arcface.yaml

# PointNeXt - Train: MS1MV3 - 10000 classes
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_single_image.py --num_points 2048 --config-path /home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/log/ms1mv3_3d_arcface/ms1mv3_3d_arcface-train-pointnext-s_arcface-ngpus1-seed345-20230511-151228-nQafgJFqP4i83ECK4VrzPg/pointnext-s_arcface.yaml


#####################################


# PointNeXt - Train: WebFace260M -  1000 classes
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_single_image.py --num_points 1024 --config-path /home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/log/webface_3d_arcface/webface_3d_arcface-train-pointnext-s_arcface-ngpus1-seed3095-20230605-195440-iwUojCxmTg7o6SNF2JHpua/pointnext-s_arcface.yaml

# PointNeXt - Train: WebFace260M -  2000 classes
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_single_image.py --num_points 1024 --config-path /home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/log/webface_3d_arcface/webface_3d_arcface-train-pointnext-s_arcface-ngpus1-seed3095-20230605-232024-ChD5VgUfMqgp6r4SsQhrzL/pointnext-s_arcface.yaml

# PointNeXt - Train: WebFace260M -  5000 classes
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_single_image.py --num_points 1024 --config-path /home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/log/webface_3d_arcface/webface_3d_arcface-train-pointnext-s_arcface-ngpus1-seed3095-20230608-104723-F7xYY2Vnp4xS6EX28h3JxA/pointnext-s_arcface.yaml

# PointNeXt - Train: WebFace260M - 10000 classes
export CUDA_VISIBLE_DEVICES=0; python examples/face_verification/eval_ijbc_single_image.py --num_points 2048 --config-path /home/bjgbiesseck/GitHub/BOVIFOCR_PointNeXt_3D_face_recognition/log/webface_3d_arcface/webface_3d_arcface-train-pointnext-s_arcface-ngpus1-seed3095-20230614-093329-a5Gtk2dX6odcuCM23vy68Y/pointnext-s_arcface.yaml
