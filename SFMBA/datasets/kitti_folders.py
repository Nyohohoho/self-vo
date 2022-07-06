import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path


def load_as_uint(path):
    return imread(path).astype(np.uint8)


class KittiRaw(data.Dataset):

    def __init__(self, root, train=True, transform=None):
        super(KittiRaw, self).__init__()

        self.root = Path(root)

        scene_list_path = self.root / 'train.txt' if train else self.root / 'val.txt'

        self.image_folder = [self.root / folder[:-1] for folder in open(scene_list_path)]
        self.intrinsics_folder = [self.root / folder[:-1] / "cam.txt" for folder in open(scene_list_path)]

        sequence_set = self.obtain_data()
        self.samples = sequence_set
        self.transform = transform

    def obtain_data(self):
        sequence_set = []
        for f_img, f_intrinsics in zip(self.image_folder, self.intrinsics_folder):
            images = sorted(f_img.files("*.jpg"))
            intrinsics = np.genfromtxt(f_intrinsics).astype(np.float32).reshape((3, 3))

            shifts = [1, 2, 3]
            for left_shift in shifts:
                for right_shift in shifts:
                    for i in range(left_shift, len(images) - right_shift):
                        sample = {'intrinsics': intrinsics, 'tgt_img': images[i], 'src_imgs': []}
                        sample['src_imgs'].append(images[i - left_shift])
                        sample['src_imgs'].append(images[i + right_shift])

                        sequence_set.append(sample)

        return sequence_set

    def __getitem__(self, item):
        sample = self.samples[item]
        intrinsics = sample['intrinsics']
        tgt_img = load_as_uint(sample['tgt_img'])
        src_imgs = [load_as_uint(src_img) for src_img in sample['src_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + src_imgs, intrinsics)
            tgt_img = imgs[0]
            src_imgs = imgs[1:]

        return tgt_img, src_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)


class KittiOdometry(data.Dataset):

    def __init__(self, root, train=True, transform=None):
        super(KittiOdometry, self).__init__()

        self.root = Path(root)

        scene_list = ['00', '03', '06', '07', '05', '01', '04', '02', '08'] if train else ['09', '10']

        self.image_folder = [self.root / folder / "image_2" for folder in scene_list]
        self.intrinsics_folder = [self.root / folder / "intrinsics.txt" for folder in scene_list]

        sequence_set = self.obtain_data()
        self.samples = sequence_set
        self.transform = transform

    def obtain_data(self):
        sequence_set = []
        for f_img, f_intrinsics in zip(self.image_folder, self.intrinsics_folder):
            images = sorted(f_img.files("*.png"))
            intrinsics = np.genfromtxt(f_intrinsics).astype(np.float32).reshape((3, 3))

            shifts = [1, 2, 3]
            for left_shift in shifts:
                for right_shift in shifts:
                    for i in range(left_shift, len(images) - right_shift):
                        sample = {'intrinsics': intrinsics, 'tgt_img': images[i], 'src_imgs': []}
                        sample['src_imgs'].append(images[i - left_shift])
                        sample['src_imgs'].append(images[i + right_shift])

                        sequence_set.append(sample)

        return sequence_set

    def __getitem__(self, item):
        sample = self.samples[item]
        intrinsics = sample['intrinsics']
        tgt_img = load_as_uint(sample['tgt_img'])
        src_imgs = [load_as_uint(src_img) for src_img in sample['src_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + src_imgs, intrinsics)
            tgt_img = imgs[0]
            src_imgs = imgs[1:]

        return tgt_img, src_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)


class KittiBundleAdjust(data.Dataset):

    def __init__(self, root, train=True, transform=None):
        super(KittiBundleAdjust, self).__init__()

        self.root = Path(root)

        scene_list = ['00', '03', '06', '07', '05', '01', '04', '02', '08'] if train else ['09', '10']

        self.image_folder = [self.root / folder / "image_2" for folder in scene_list]
        self.intrinsics_folder = [self.root / folder / "intrinsics.txt" for folder in scene_list]
        self.sequence_length = 28

        sequence_set = self.obtain_data()
        self.samples = sequence_set
        self.transform = transform

    def obtain_data(self):
        sequence_set = []
        for f_img, f_intrinsics in zip(self.image_folder, self.intrinsics_folder):
            images = sorted(f_img.files("*.png"))
            intrinsics = np.genfromtxt(f_intrinsics).astype(np.float32).reshape((3, 3))

            for i in range(0, len(images) - self.sequence_length + 1):
                frames = []
                for j in range(0, self.sequence_length):
                    frames.append(images[i + j])
                sample = {'intrinsics': intrinsics, 'frames': frames}
                sequence_set.append(sample)

        return sequence_set

    def __getitem__(self, item):
        sample = self.samples[item]
        intrinsics = sample['intrinsics']
        frames = [load_as_uint(frame_path) for frame_path in sample['frames']]
        if self.transform is not None:
            frames, intrinsics = self.transform(frames, intrinsics)

        return frames, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
