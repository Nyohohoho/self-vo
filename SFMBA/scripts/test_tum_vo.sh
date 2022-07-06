DATASET_DIR=/home/don/Dataset/tum_rgbd/test/
OUTPUT_DIR=/home/don/Projects/SFMBA/checkpoints/tum/07-05-19:31/results/
POSE_NET=/home/don/Projects/SFMBA/checkpoints/tum/07-05-19:31/pose_ckpt.tar

python test_tum.py \
--img-height 192 --img-width 256 \
--sequence fr2/desk \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR
