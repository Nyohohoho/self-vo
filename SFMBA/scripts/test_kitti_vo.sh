DATASET_DIR=/home/don/Dataset/kitti/sequences/
OUTPUT_DIR=/home/don/Projects/SFMBA/checkpoints/maxpool/06-01-17:43/results/
POSE_NET=/home/don/Projects/SFMBA/checkpoints/maxpool/06-01-17:43/pose_ckpt.tar

python test_vo.py \
--img-height 256 --img-width 512 \
--sequence 09 \
--pretrained-posenet $POSE_NET \
--dataset-dir $DATASET_DIR \
--output-dir $OUTPUT_DIR

python test_vo.py \
--img-height 256 --img-width 512 \
--sequence 10 \
--pretrained-posenet $POSE_NET \
--dataset-dir $DATASET_DIR \
--output-dir $OUTPUT_DIR

python ./kitti_eval/eval_odom.py --result=$OUTPUT_DIR --align='7dof'
