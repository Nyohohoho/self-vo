TRAIN_SET=/home/don/Dataset/tum_rgbd/

python train_two_view.py $TRAIN_SET \
-b4 \
--lr 1e-4 \
--wp 1.0 --ws 1e-3 --wr 1e-5 \
--img-height 192 --img-width 256 \
--epochs 20 \
--name tum
