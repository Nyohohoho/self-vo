import torch

from imageio import imread, imsave
from skimage.transform import resize as imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R

import models
import utils

import cv2

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-posenet", required=True, type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=192, type=int, help="Image height")
parser.add_argument("--img-width", default=256, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--output-dir", type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)

parser.add_argument("--sequence", default='09', type=str, help="sequence to test")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def load_tensor_image(image_filename, args):
    img = imread(image_filename).astype(np.float32)

    h, w, _ = img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = normalize(
        torch.from_numpy(img).to(device) / 255.0,
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    ).unsqueeze(0)

    return tensor_img


@torch.no_grad()
def main():
    args = parser.parse_args()

    weights_pose = torch.load(args.pretrained_posenet, map_location=device)
    pose_net = models.PoseNet().to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
    pose_net.eval()

    image_dir = Path(args.dataset_dir + args.sequence + "/rgb/")
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    test_files = sum([image_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])
    test_files.sort()

    print('{} files to test'.format(len(test_files)))

    n = len(test_files)
    tensor_img1 = load_tensor_image(test_files[0], args)
    time_stamp = np.array(test_files[0].replace(image_dir, '').replace('.png', '')).reshape(-1)

    global_pose = np.eye(4)
    translation = global_pose[:3, 3].reshape(-1)
    rotation = R.from_matrix(global_pose[:3, :3]).as_quat().reshape(-1)
    poses = [np.concatenate([time_stamp, translation, rotation]).reshape(1, -1)]

    for iter in tqdm(range(n - 1)):

        tensor_img2 = load_tensor_image(test_files[iter + 1], args)
        time_stamp = np.array(test_files[iter + 1].replace(image_dir, '').replace('.png', '')).reshape(-1)

        pose, _ = pose_net(tensor_img1, tensor_img2)

        pose_mat = utils.pose_vec2mat(pose).squeeze(0).cpu().numpy()
        pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])

        global_pose = global_pose @ pose_mat
        translation = global_pose[:3, 3].reshape(-1)
        rotation = R.from_matrix(global_pose[:3, :3]).as_quat().reshape(-1)
        poses.append(np.concatenate([time_stamp, translation, rotation]).reshape(1, -1))

        # update
        tensor_img1 = tensor_img2

    poses = np.concatenate(poses, axis=0)
    filename = Path(args.output_dir + args.sequence.replace('/', '_') + ".txt")
    np.savetxt(filename, poses, delimiter=' ', fmt='%s')


if __name__ == '__main__':
    main()
