DATASET_TYPE=tum_rgbd
DATASET_DIR=/home/don/Dataset/tum_rgbd/

DEPTH=/home/don/Projects/SFMBA/checkpoints/tum/07-05-19:31/depth_ckpt.tar

python -W ignore visualize.py \
--img-height 192 --img-width 256 \
--sequence 08 \
--pretrained-depth $DEPTH \
--dataset-type $DATASET_TYPE \
--dataset-dir $DATASET_DIR
