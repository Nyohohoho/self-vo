import torch
import torch.backends.cuda
import numpy as np
import time
import argparse

from imageio import imread
from skimage.transform import resize as imresize
from path import Path
from tqdm import tqdm

import models
import utils

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-posenet", required=True, type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=192, type=int, help="image height")
parser.add_argument("--img-width", default=640, type=int, help="image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--output-dir", type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--sequence", default='09', type=str, help="sequence to test")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.backends.cuda.matmul.allow_tf32 = False


def load_tensor_image(image_filename, args):
    global device

    img = imread(image_filename).astype(np.float32)
    h, w, _ = img.shape

    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)

    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).to(device) / 255.0 - 0.5) / 0.5).unsqueeze(0)

    return tensor_img


@torch.no_grad()
def main():
    args = parser.parse_args()

    weights_pose = torch.load(args.pretrained_posenet, map_location=device)
    pose_net = models.PoseNet().to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
    pose_net.eval()

    image_dir = Path(args.dataset_dir + args.sequence + "/image_2/")
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    test_files = sum([image_dir.files('*.{}'.format(ext)) for ext in ['png', 'jpg', 'bmp']], [])
    test_files.sort()

    print('{} files to test'.format(len(test_files)))

    global_pose = np.eye(4)
    poses = [global_pose[0:3, :].reshape(1, 12)]

    n = len(test_files)
    tensor_img1 = load_tensor_image(test_files[0], args)

    avg_time = 0
    for iter in tqdm(range(n - 1)):

        tensor_img2 = load_tensor_image(test_files[iter + 1], args)

        # torch.cuda.synchronize()
        t_start = time.time()

        pose, _ = pose_net(tensor_img1, tensor_img2)

        # torch.cuda.synchronize()
        elapsed_time = time.time() - t_start

        avg_time += elapsed_time

        pose_mat = utils.pose_vec2mat(pose)
        pose_mat = pose_mat.squeeze(0).cpu().numpy()
        pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
        global_pose = global_pose @ np.linalg.inv(pose_mat)
        poses.append(global_pose[0:3, :].reshape(1, 12))

        # update
        tensor_img1 = tensor_img2

    poses = np.concatenate(poses, axis=0)
    filename = Path(args.output_dir + args.sequence + ".txt")
    np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')

    avg_time /= n
    print('Avg Time:  ', avg_time * 1000, ' ms')
    print('Avg Speed: ', 1.0 / avg_time, ' fps')


if __name__ == '__main__':
    main()
