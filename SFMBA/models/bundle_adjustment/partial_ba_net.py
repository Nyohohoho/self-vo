import torch.nn as nn

from .partial_resnet_encoder_3d import PartialResNetEncoder
from .partial_depth_decoder_3d import PartialDepthDecoder
from .partial_pose_decoder_3d import PartialPoseDecoder


class PartialBANet(nn.Module):
    def __init__(self):
        super(PartialBANet, self).__init__()
        self.encoder = PartialResNetEncoder()
        self.depth_decoder = PartialDepthDecoder()
        self.pose_decoder = PartialPoseDecoder()

    def forward(self, x, x_mask):
        features, masks = self.encoder(x, x_mask)

        depth = self.depth_decoder(features, masks)
        poses = self.pose_decoder(features, masks)

        return depth, poses


if __name__ == '__main__':
    import torch
    import torch.cuda
    from time import time

    net = PartialBANet().cuda()
    image = torch.rand(1, 3, 4, 192, 640).cuda()
    mask = (torch.rand(1, 3, 4, 192, 640).cuda() > 0.5).float()

    for i in range(100):
        torch.cuda.synchronize(0)
        t_start = time()

        depth, poses = net(image, mask)
        print("depth", depth.shape)
        print("poses", [pose.shape for pose in poses])

        torch.cuda.synchronize(0)
        print((time() - t_start) * 1000, "ms")
