import torch
from skimage.transform import resize as imresize
from imageio import imread
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import models
import time

parser = argparse.ArgumentParser(description='Script for DepthNEt testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-depthnet", required=True, type=str, help="pretrained depth net path")
parser.add_argument("--img-height", default=192, type=int, help="Image height")
parser.add_argument("--img-width", default=640, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--output-dir", default=None, required=True, type=str, help="Output directory for saving predictions in a big 3D numpy file")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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

    weights_depth = torch.load(args.pretrained_depthnet, map_location=device)
    depth_net = models.DepthResNet().to(device)
    depth_net.load_state_dict(weights_depth['state_dict'], strict=False)
    depth_net.eval()

    dataset_dir = Path(args.dataset_dir)

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = list(f.read().splitlines())
    else:
        test_files = sorted(dataset_dir.files('*.png'))

    print('{} files to test'.format(len(test_files)))
  
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    avg_time = 0
    predictions = []
    for iter in tqdm(range(len(test_files))):
        tensor_img = load_tensor_image(test_files[iter], args)

        # compute speed
        torch.cuda.synchronize()
        t_start = time.time()

        output = depth_net(tensor_img, None)[0]

        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        
        avg_time += elapsed_time

        pred_depth = output.squeeze().cpu().numpy()

        predictions.append(pred_depth)

    predictions = np.array(predictions)
    np.save(output_dir/'predictions.npy', predictions)

    avg_time /= len(test_files)
    print('Avg Time: ', avg_time * 1000, ' ms')
    print('Avg Speed: ', 1.0 / avg_time, ' fps')


if __name__ == '__main__':
    main()
