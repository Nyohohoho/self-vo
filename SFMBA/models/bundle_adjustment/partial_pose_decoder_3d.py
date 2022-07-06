import torch
import torch.nn as nn
from ..layers import PartialConv3d, PartialConvBlock


class PartialPoseDecoder(nn.Module):

    def __init__(self):
        super(PartialPoseDecoder, self).__init__()

        # The number of channels used in ResNet encoder
        encoder_channels = [64, 64, 128, 256, 512]

        # The number of channels used in decoder (You can modify)
        decoder_channels = [256, 256]

        output_channel = 6

        self.regress_conv1 = PartialConvBlock(
            conv_layer=PartialConv3d(encoder_channels[4], decoder_channels[0],
                                     kernel_size=3, stride=1, padding=1,
                                     bias=False, multi_channel=True, return_mask=True),
            activation=nn.ReLU(inplace=True)
        )

        self.regress_conv2 = PartialConvBlock(
            conv_layer=PartialConv3d(decoder_channels[0], decoder_channels[1],
                                     kernel_size=3, stride=1, padding=1,
                                     bias=False, multi_channel=True, return_mask=True),
            activation=nn.ReLU(inplace=True)
        )

        self.predict_output = PartialConv3d(decoder_channels[1], output_channel,
                                            kernel_size=1, stride=1, padding=0,
                                            bias=False, multi_channel=True, return_mask=True)

    def forward(self, features, masks):
        encoded_last = features[-1]
        mask_last = masks[-1]
        
        out_regress_conv1, out_regress_mask1 = self.regress_conv1(encoded_last, mask_last)
        out_regress_conv2, out_regress_mask2 = self.regress_conv2(out_regress_conv1, out_regress_mask1)

        basis_poses, _ = self.predict_output(out_regress_conv2, out_regress_mask2)
        aggregated_poses = torch.mean(0.01 * basis_poses, dim=[3, 4])
        poses = [pose.view(aggregated_poses.shape[0], 6)
                 for pose in torch.chunk(aggregated_poses, chunks=aggregated_poses.shape[2], dim=2)]

        return poses
