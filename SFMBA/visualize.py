import torch
import torch.cuda

import numpy as np
import argparse

import matplotlib.colors as colors
import matplotlib.cm as cm
from skimage.transform import resize as imresize
from path import Path
from time import time
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from scipy.ndimage import map_coordinates

import models

import cv2

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-depth", required=True, type=str, help="pretrained encoder path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--dataset-type", type=str, help="Dataset type")
parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

parser.add_argument("--sequence", default='09', type=str, help="sequence to test")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_lut(lut_path):
    lut = np.fromfile(lut_path, np.double)
    lut = lut.reshape([2, lut.size // 2])
    bilinear_lut = lut.transpose()

    return bilinear_lut


def undistort(image, bilinear_lut):
    lut = bilinear_lut[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))
    undistorted = np.rollaxis(np.array([map_coordinates(image[:, :, channel], lut, order=1)
                                        for channel in range(0, image.shape[2])]), 0, 3)

    return undistorted.astype(image.dtype)


def load_image(image_filename, args):
    raw_image = Image.open(image_filename)
    if args.dataset_type == 'robotcar':
        raw_image = demosaic(raw_image, 'gbrg')
        lut_path = Path(args.dataset_dir + "/distortion_lut.bin")
        bilinear_lut = load_lut(lut_path)
        raw_image = undistort(raw_image, bilinear_lut)

    numpy_img = np.array(raw_image, dtype=np.uint8)
    if args.dataset_type == 'euroc_mav':
        numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_GRAY2RGB)

    if args.dataset_type == 'robotcar' or args.dataset_type == 'cityscapes':
        # Crop the unnecessary part of car
        h, w, _ = numpy_img.shape
        numpy_img = numpy_img[:round(h * 0.8), :]

    h, w, _ = numpy_img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = imresize(numpy_img.astype(np.float32), (args.img_height, args.img_width))
    else:
        img = numpy_img
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).to(device) / 255.0 - 0.5) / 0.5).unsqueeze(0)

    return cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR), tensor_img


def colorize(raw_map):
    normalizer = colors.Normalize(vmin=raw_map.min(), vmax=np.percentile(raw_map, q=95))
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    color_map = (mapper.to_rgba(raw_map)[:, :, :3] * 255).astype(np.uint8)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)

    return color_map


@torch.no_grad()
def main():
    args = parser.parse_args()

    weights_depth = torch.load(args.pretrained_depth, map_location=device)
    depth_resnet = models.DepthResNet().to(device)
    depth_resnet.load_state_dict(weights_depth['state_dict'], strict=False)
    depth_resnet.eval()

    if args.dataset_type == 'kitti':
        image_dir = Path(args.dataset_dir + "sequences/" + args.sequence + "/image_2/")
        # intrinsics_file = Path(args.dataset_dir + "sequences/" + args.sequence + "/intrinsics.txt")
    elif args.dataset_type == 'robotcar':
        image_dir = Path(args.dataset_dir + "partial")
        # intrinsics_file = Path(args.dataset_dir + "/intrinsics.txt")
    else:
        image_dir = Path(args.dataset_dir + "sample")
        # intrinsics_file = Path(args.dataset_dir + "/intrinsics.txt")

    test_files = sum([image_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])
    test_files.sort()

    print('{} files to test'.format(len(test_files)))

    n = len(test_files)
    h, w, _ = cv2.imread(test_files[0], cv2.IMREAD_COLOR).shape
    if args.dataset_type == 'robotcar' or args.dataset_type == 'cityscapes':
        # Shrink output size
        h, w = round(h / 2), round(w / 2)

    if args.dataset_type == 'kitti':
        video_writer = cv2.VideoWriter(
            'visualization/results/video/output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (w, int(h * 2))
        )
    else:
        video_writer = cv2.VideoWriter(
            'visualization/results/video/output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (int(w * 2), h)
        )

    time_recorder = []
    for i in range(n):
        tgt_numpy_img, tgt_tensor_img = load_image(test_files[i], args)

        torch.cuda.synchronize()
        start_time = time()

        depth = depth_resnet(tgt_tensor_img)

        torch.cuda.synchronize()
        end_time = time()

        # million second
        elapsed_time = (end_time - start_time) * 1000
        print("Current Progress: {:04d}/{:04d}".format((i + 1), n), "Runtime: {:2f}".format(elapsed_time), "ms")
        time_recorder.append(elapsed_time)

        # output = depth
        output = 1.0 / depth[0]

        output = output.squeeze().cpu().numpy()
        output = cv2.resize(output, (w, h))
        color_output = colorize(output)

        if args.dataset_type == 'kitti':
            comparison = cv2.vconcat([tgt_numpy_img, color_output])
        else:
            comparison = cv2.hconcat([tgt_numpy_img, color_output])

        cv2.imshow("com", comparison)
        cv2.waitKey(1)

        cv2.imwrite("visualization/results/image/{:06d}.png".format(i), comparison)
        video_writer.write(comparison)

    average_time = sum(time_recorder) / len(time_recorder)
    print("Average RunTime:", average_time, "ms")
    video_writer.release()


if __name__ == '__main__':
    main()
