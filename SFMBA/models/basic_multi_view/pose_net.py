import torch
import torch.nn as nn
from .resnet_encoder import ResNetEncoder
from ..layers import ConvBlock
pose_channel = 6
affine_channel = 1


class PoseNet(nn.Module):

    def __init__(self):
        super(PoseNet, self).__init__()

        input_plane = 2 * 3
        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_plane, conv_planes[0],
                      kernel_size=7, padding=3, stride=2, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_planes[0], conv_planes[1],
                      kernel_size=5, padding=2, stride=2, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv_planes[1], conv_planes[2],
                      kernel_size=3, padding=1, stride=2, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(conv_planes[2], conv_planes[3],
                      kernel_size=3, padding=1, stride=2, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(conv_planes[3], conv_planes[4],
                      kernel_size=3, padding=1, stride=2, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(conv_planes[4], conv_planes[5],
                      kernel_size=3, padding=1, stride=2, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(conv_planes[5], conv_planes[6],
                      kernel_size=3, padding=1, stride=2, bias=False),
            nn.ReLU(inplace=True)
        )

        self.predict_pose = nn.Conv2d(conv_planes[6], pose_channel,
                                      kernel_size=1, stride=1, padding=0, bias=False)
        self.predict_affine_a = nn.Sequential(
            nn.Conv2d(conv_planes[6], affine_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.Softplus()
        )
        self.predict_affine_b = nn.Sequential(
            nn.Conv2d(conv_planes[6], affine_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, tgt_img, src_img):
        x = torch.cat([tgt_img, src_img], dim=1)
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        avg_conv = torch.mean(out_conv7, dim=[2, 3], keepdim=True)

        pose = self.predict_pose(avg_conv)
        pose = 0.01 * pose.view(pose.shape[0], pose_channel)

        affine_a = self.predict_affine_a(avg_conv)
        affine_b = self.predict_affine_b(avg_conv)
        affine_param = [affine_a, affine_b]

        return pose, affine_param


class PoseResNet(nn.Module):

    def __init__(self):
        super(PoseResNet, self).__init__()

        self.pose_encoder = ResNetEncoder(num_of_inputs=2)
        # The number of channels used in ResNet encoder
        encoder_channels = [64, 64, 128, 256, 512]

        # The number of channels used in decoder
        decoder_channels = [256, 256, 256]

        pose_channel = 6
        affine_channel = 1

        self.squeeze_conv = nn.Conv2d(encoder_channels[4], decoder_channels[0],
                                      kernel_size=1, stride=1, padding=0, bias=False)

        self.regress_conv1 = ConvBlock(
            conv_layer=nn.Conv2d(decoder_channels[0], decoder_channels[1],
                                 kernel_size=3, stride=1, padding=1, bias=False),
            activation=nn.ReLU(inplace=True)
        )

        self.regress_conv2 = ConvBlock(
            conv_layer=nn.Conv2d(decoder_channels[1], decoder_channels[2],
                                 kernel_size=3, stride=1, padding=1, bias=False),
            activation=nn.ReLU(inplace=True)
        )

        self.predict_pose = nn.Conv2d(decoder_channels[2], pose_channel,
                                      kernel_size=1, stride=1, padding=0, bias=False)
        self.predict_affine_a = nn.Sequential(
            nn.Conv2d(decoder_channels[2], affine_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.Softplus()
        )
        self.predict_affine_b = nn.Sequential(
            nn.Conv2d(decoder_channels[2], affine_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, tgt_img, src_img):
        concat_img = torch.cat([tgt_img, src_img], dim=1)
        features = self.pose_encoder(concat_img)
        last_feat = features[-1]

        squeezed_feat = self.squeeze_conv(last_feat)
        out_regress_conv1 = self.regress_conv1(squeezed_feat)
        out_regress_conv2 = self.regress_conv2(out_regress_conv1)

        pose = self.predict_pose(out_regress_conv2)
        pose = torch.mean(0.01 * pose, dim=[2, 3])

        affine_a = self.predict_affine_a(out_regress_conv2)
        affine_a = torch.mean(affine_a, dim=[2, 3], keepdim=True)
        affine_b = self.predict_affine_b(out_regress_conv2)
        affine_b = torch.mean(affine_b, dim=[2, 3], keepdim=True)
        affine_param = [affine_a, affine_b]

        return pose, affine_param
