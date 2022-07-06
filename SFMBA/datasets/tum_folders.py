import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path


def load_as_uint(path):
    return imread(path).astype(np.uint8)


class TUMSequence(data.Dataset):

    def __init__(self, root, train=True, transform=None):
        if train:
            self.root = Path(root) / 'train'
        else:
            self.root = Path(root) / 'test'

        scene_list_path = self.root / 'list.txt'

        self.image_folder = [self.root / folder[:-1] for folder in open(scene_list_path)]
        self.intrinsics_folder = [self.root / folder[:-1] / "intrinsics.txt" for folder in open(scene_list_path)]

        sequence_set = self.obtain_data()
        self.samples = sequence_set
        self.transform = transform

    def obtain_data(self):
        sequence_set = []
        for f_img, f_intrinsics in zip(self.image_folder, self.intrinsics_folder):
            images = sorted((f_img / 'rgb').files("*.png"))
            intrinsics = np.genfromtxt(f_intrinsics).astype(np.float32).reshape((3, 3))

            for i in range(1, len(images) - 1):
                sample = {'intrinsics': intrinsics, 'tgt_img': images[i], 'src_imgs': []}
                sample['src_imgs'].append(images[i - 1])
                sample['src_imgs'].append(images[i + 1])

                sequence_set.append(sample)

        return sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        intrinsics = sample['intrinsics']
        tgt_img = load_as_uint(sample['tgt_img'])
        src_imgs = [load_as_uint(src_img) for src_img in sample['src_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + src_imgs, np.copy(intrinsics))
            tgt_img = imgs[0]
            src_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])

        return tgt_img, src_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)