DATA_ROOT=/home/don/Dataset/kitti/raw/kitti_depth_test
RESULTS_DIR=results/test

DEPTH=/home/don/Projects/SFMBA/checkpoints/basic_two_view/05-31-11:51/depth_ckpt.tar

# test
python test_depth.py \
--img-height 256 --img-width 512 \
--pretrained-depthnet $DEPTH \
--dataset-dir $DATA_ROOT/color \
--output-dir $RESULTS_DIR

# evaluate
python ./kitti_eval/eval_depth.py \
--dataset kitti \
--pred_depth=$RESULTS_DIR/predictions.npy \
--gt_depth=$DATA_ROOT/depth
