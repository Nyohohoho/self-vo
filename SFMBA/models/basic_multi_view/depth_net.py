import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_encoder import ResNetEncoder
from ..layers import UpsampleBlock, ConvBlock


class DepthResNet(nn.Module):

    def __init__(self):
        super(DepthResNet, self).__init__()

        self.depth_encoder = ResNetEncoder()

        # The number of channels used in ResNet encoder
        encoder_channels = [64, 64, 128, 256, 512]

        # The number of channels used in decoder (You can modify)
        decoder_channels = [256, 128, 64, 32, 16]

        output_channel = 1

        # Level 5
        self.up_conv5 = ConvBlock(
            conv_layer=nn.Conv2d(encoder_channels[4], decoder_channels[0],
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.ELU(inplace=True),
            interpolation=UpsampleBlock(scale_factor=2, mode='nearest')
        )
        self.integral_conv5 = ConvBlock(
            conv_layer=nn.Conv2d(encoder_channels[3] + decoder_channels[0], decoder_channels[0],
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.ELU(inplace=True)
        )

        # Level 4
        self.up_conv4 = ConvBlock(
            conv_layer=nn.Conv2d(decoder_channels[0], decoder_channels[1],
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.ELU(inplace=True),
            interpolation=UpsampleBlock(scale_factor=2, mode='nearest')
        )
        self.integral_conv4 = ConvBlock(
            conv_layer=nn.Conv2d(encoder_channels[2] + decoder_channels[1], decoder_channels[1],
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.ELU(inplace=True)
        )
        self.predict_depth4 = ConvBlock(
            conv_layer=nn.Conv2d(decoder_channels[1], output_channel,
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.Softplus()
        )
        self.predict_uncertainty4 = ConvBlock(
            conv_layer=nn.Conv2d(decoder_channels[1], output_channel,
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.Sigmoid()
        )

        # Level 3
        self.up_conv3 = ConvBlock(
            conv_layer=nn.Conv2d(decoder_channels[1], decoder_channels[2],
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.ELU(inplace=True),
            interpolation=UpsampleBlock(scale_factor=2, mode='nearest')
        )
        self.integral_conv3 = ConvBlock(
            conv_layer=nn.Conv2d(encoder_channels[1] + decoder_channels[2], decoder_channels[2],
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.ELU(inplace=True)
        )
        self.predict_depth3 = ConvBlock(
            conv_layer=nn.Conv2d(decoder_channels[2], output_channel,
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.Softplus()
        )
        self.predict_uncertainty3 = ConvBlock(
            conv_layer=nn.Conv2d(decoder_channels[2], output_channel,
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.Sigmoid()
        )

        # Level 2
        self.up_conv2 = ConvBlock(
            conv_layer=nn.Conv2d(decoder_channels[2], decoder_channels[3],
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.ELU(inplace=True),
            interpolation=UpsampleBlock(scale_factor=2, mode='nearest')
        )
        self.integral_conv2 = ConvBlock(
            conv_layer=nn.Conv2d(encoder_channels[0] + decoder_channels[3], decoder_channels[3],
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.ELU(inplace=True)
        )
        self.predict_depth2 = ConvBlock(
            conv_layer=nn.Conv2d(decoder_channels[3], output_channel,
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.Softplus()
        )
        self.predict_uncertainty2 = ConvBlock(
            conv_layer=nn.Conv2d(decoder_channels[3], output_channel,
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.Sigmoid()
        )

        # Level 1
        self.up_conv1 = ConvBlock(
            conv_layer=nn.Conv2d(decoder_channels[3], decoder_channels[4],
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.ELU(inplace=True),
            interpolation=UpsampleBlock(scale_factor=2, mode='nearest')
        )
        self.integral_conv1 = ConvBlock(
            conv_layer=nn.Conv2d(decoder_channels[4], decoder_channels[4],
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.ELU(inplace=True)
        )
        self.predict_depth1 = ConvBlock(
            conv_layer=nn.Conv2d(decoder_channels[4], output_channel,
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.Softplus()
        )
        self.predict_uncertainty1 = ConvBlock(
            conv_layer=nn.Conv2d(decoder_channels[4], output_channel,
                                 kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            activation=nn.Sigmoid()
        )

    def _crop_like(self, larger_tensor, expected_tensor):
        assert (larger_tensor.shape[2] >= expected_tensor.shape[2]
                and larger_tensor.shape[3] >= expected_tensor.shape[3])
        return larger_tensor[:, :, :expected_tensor.shape[2], :expected_tensor.shape[3]]

    def _scale(self, input_tensor, h, w):
        if input_tensor.shape[2] < h or input_tensor.shape[3] < w:
            output_tensor = F.interpolate(input_tensor, (h, w), mode='nearest')
        else:
            output_tensor = input_tensor

        return output_tensor

    def _normalize(self, input_tensor):
        input_mean = torch.mean(input_tensor, dim=[2, 3], keepdim=True)
        output_tensor = input_tensor / input_mean
        return output_tensor

    def forward(self, img):
        _, _, h, w = img.shape
        features = self.depth_encoder(img)
        encoded1, encoded2, encoded3, encoded4, encoded5 = features

        out_up_conv5 = self.up_conv5(encoded5)
        out_up_conv5 = self._crop_like(out_up_conv5, encoded4)
        concat5 = torch.cat([out_up_conv5, encoded4], dim=1)
        out_integral_conv5 = self.integral_conv5(concat5)

        out_up_conv4 = self.up_conv4(out_integral_conv5)
        out_up_conv4 = self._crop_like(out_up_conv4, encoded3)
        concat4 = torch.cat([out_up_conv4, encoded3], dim=1)
        out_integral_conv4 = self.integral_conv4(concat4)

        out_up_conv3 = self.up_conv3(out_integral_conv4)
        out_up_conv3 = self._crop_like(out_up_conv3, encoded2)
        concat3 = torch.cat([out_up_conv3, encoded2], dim=1)
        out_integral_conv3 = self.integral_conv3(concat3)

        out_up_conv2 = self.up_conv2(out_integral_conv3)
        out_up_conv2 = self._crop_like(out_up_conv2, encoded1)
        concat2 = torch.cat([out_up_conv2, encoded1], dim=1)
        out_integral_conv2 = self.integral_conv2(concat2)

        out_up_conv1 = self.up_conv1(out_integral_conv2)
        concat1 = torch.cat([out_up_conv1], dim=1)
        out_integral_conv1 = self.integral_conv1(concat1)

        if self.training:
            predicted_depth1 = self.predict_depth1(out_integral_conv1)
            predicted_uncertainty1 = self.predict_uncertainty1(out_integral_conv1)
            predicted_depth2 = self.predict_depth2(out_integral_conv2)
            predicted_uncertainty2 = self.predict_uncertainty2(out_integral_conv2)
            predicted_depth3 = self.predict_depth3(out_integral_conv3)
            predicted_uncertainty3 = self.predict_uncertainty3(out_integral_conv3)
            predicted_depth4 = self.predict_depth4(out_integral_conv4)
            predicted_uncertainty4 = self.predict_uncertainty4(out_integral_conv4)

            normalized_depth1 = self._normalize(self._scale(predicted_depth1, h, w))
            normalized_depth2 = self._normalize(self._scale(predicted_depth2, h, w))
            normalized_depth3 = self._normalize(self._scale(predicted_depth3, h, w))
            normalized_depth4 = self._normalize(self._scale(predicted_depth4, h, w))

            scaled_uncertainty1 = self._scale(predicted_uncertainty1, h, w)
            scaled_uncertainty2 = self._scale(predicted_uncertainty2, h, w)
            scaled_uncertainty3 = self._scale(predicted_uncertainty3, h, w)
            scaled_uncertainty4 = self._scale(predicted_uncertainty4, h, w)

            return [(normalized_depth1, scaled_uncertainty1), (normalized_depth2, scaled_uncertainty2),
                    (normalized_depth3, scaled_uncertainty3), (normalized_depth4, scaled_uncertainty4)]

        else:
            predicted_depth = self.predict_depth1(out_integral_conv1)
            normalized_depth = self._normalize(self._scale(predicted_depth, h, w))

            return normalized_depth
