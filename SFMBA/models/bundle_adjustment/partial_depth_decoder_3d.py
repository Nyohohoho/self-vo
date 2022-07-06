import torch
import torch.nn as nn
from ..layers import PartialConv3d, UpsampleBlock, PartialConvBlock


class PartialDepthDecoder(nn.Module):

    def __init__(self):
        super(PartialDepthDecoder, self).__init__()

        # The number of channels used in ResNet encoder
        encoder_channels = [64, 64, 128, 256, 512]

        # The number of channels used in decoder (You can modify)
        decoder_channels = [256, 128, 64, 32, 16]

        output_channel = 1

        self.up_conv5 = PartialConvBlock(
            conv_layer=PartialConv3d(encoder_channels[4], decoder_channels[0],
                                     kernel_size=3, stride=1, padding=1,
                                     bias=False, multi_channel=True, return_mask=True),
            activation=nn.ELU(inplace=True),
            interpolation=UpsampleBlock(scale_factor=(1, 2, 2), mode='nearest')
        )
        self.integral_conv5 = PartialConvBlock(
            conv_layer=PartialConv3d(encoder_channels[3] + decoder_channels[0], decoder_channels[0],
                                     kernel_size=3, stride=1, padding=1,
                                     bias=False, multi_channel=True, return_mask=True),
            activation=nn.ELU(inplace=True)
        )

        self.up_conv4 = PartialConvBlock(
            conv_layer=PartialConv3d(decoder_channels[0], decoder_channels[1],
                                     kernel_size=3, stride=1, padding=1,
                                     bias=False, multi_channel=True, return_mask=True),
            activation=nn.ELU(inplace=True),
            interpolation=UpsampleBlock(scale_factor=(1, 2, 2), mode='nearest')
        )
        self.integral_conv4 = PartialConvBlock(
            conv_layer=PartialConv3d(encoder_channels[2] + decoder_channels[1], decoder_channels[1],
                                     kernel_size=3, stride=1, padding=1,
                                     bias=False, multi_channel=True, return_mask=True),
            activation=nn.ELU(inplace=True)
        )

        self.up_conv3 = PartialConvBlock(
            conv_layer=PartialConv3d(decoder_channels[1], decoder_channels[2],
                                     kernel_size=3, stride=1, padding=1,
                                     bias=False, multi_channel=True, return_mask=True),
            activation=nn.ELU(inplace=True),
            interpolation=UpsampleBlock(scale_factor=(1, 2, 2), mode='nearest')
        )
        self.integral_conv3 = PartialConvBlock(
            conv_layer=PartialConv3d(encoder_channels[1] + decoder_channels[2], decoder_channels[2],
                                     kernel_size=3, stride=1, padding=1,
                                     bias=False, multi_channel=True, return_mask=True),
            activation=nn.ELU(inplace=True)
        )

        self.up_conv2 = PartialConvBlock(
            conv_layer=PartialConv3d(decoder_channels[2], decoder_channels[3],
                                     kernel_size=3, stride=1, padding=1,
                                     bias=False, multi_channel=True, return_mask=True),
            activation=nn.ELU(inplace=True),
            interpolation=UpsampleBlock(scale_factor=(1, 2, 2), mode='nearest')
        )
        self.integral_conv2 = PartialConvBlock(
            conv_layer=PartialConv3d(encoder_channels[0] + decoder_channels[3], decoder_channels[3],
                                     kernel_size=3, stride=1, padding=1,
                                     bias=False, multi_channel=True, return_mask=True),
            activation=nn.ELU(inplace=True)
        )

        self.up_conv1 = PartialConvBlock(
            conv_layer=PartialConv3d(decoder_channels[3], decoder_channels[4],
                                     kernel_size=3, stride=1, padding=1,
                                     bias=False, multi_channel=True, return_mask=True),
            activation=nn.ELU(inplace=True),
            interpolation=UpsampleBlock(scale_factor=(1, 2, 2), mode='nearest')
        )
        self.integral_conv1 = PartialConvBlock(
            conv_layer=PartialConv3d(decoder_channels[4], decoder_channels[4],
                                     kernel_size=3, stride=1, padding=1,
                                     bias=False, multi_channel=True, return_mask=True),
            activation=nn.ELU(inplace=True)
        )

        self.predict_output = PartialConvBlock(
            conv_layer=PartialConv3d(decoder_channels[4], output_channel,
                                     kernel_size=3, stride=1, padding=1,
                                     bias=False, multi_channel=True, return_mask=True),
            activation=nn.Softplus()
        )

    def _crop_like(self, larger_tensor, expected_tensor):
        assert (larger_tensor.shape[3] >= expected_tensor.shape[3]
                and larger_tensor.shape[4] >= expected_tensor.shape[4])
        return larger_tensor[:, :, :, :expected_tensor.shape[3], :expected_tensor.shape[4]]

    def forward(self, features, masks):
        encoded1, encoded2, encoded3, encoded4, encoded5 = features
        mask1, mask2, mask3, mask4, mask5 = masks

        out_up_conv5, out_up_mask5 = self.up_conv5(encoded5, mask5)
        out_up_conv5 = self._crop_like(out_up_conv5, encoded4)
        out_up_mask5 = self._crop_like(out_up_mask5, mask4)
        concat5 = torch.cat([out_up_conv5, encoded4], dim=1)
        concat_mask5 = torch.cat([out_up_mask5, mask4], dim=1)
        out_integral_conv5, out_integral_mask5 = self.integral_conv5(concat5, concat_mask5)

        out_up_conv4, out_up_mask4 = self.up_conv4(out_integral_conv5, out_integral_mask5)
        out_up_conv4 = self._crop_like(out_up_conv4, encoded3)
        out_up_mask4 = self._crop_like(out_up_mask4, mask3)
        concat4 = torch.cat([out_up_conv4, encoded3], dim=1)
        concat_mask4 = torch.cat([out_up_mask4, mask3], dim=1)
        out_integral_conv4, out_integral_mask4 = self.integral_conv4(concat4, concat_mask4)

        out_up_conv3, out_up_mask3 = self.up_conv3(out_integral_conv4, out_integral_mask4)
        out_up_conv3 = self._crop_like(out_up_conv3, encoded2)
        out_up_mask3 = self._crop_like(out_up_mask3, mask2)
        concat3 = torch.cat([out_up_conv3, encoded2], dim=1)
        concat_mask3 = torch.cat([out_up_mask3, mask2], dim=1)
        out_integral_conv3, out_integral_mask3 = self.integral_conv3(concat3, concat_mask3)

        out_up_conv2, out_up_mask2 = self.up_conv2(out_integral_conv3, out_integral_mask3)
        out_up_conv2 = self._crop_like(out_up_conv2, encoded1)
        out_up_mask2 = self._crop_like(out_up_mask2, mask1)
        concat2 = torch.cat([out_up_conv2, encoded1], dim=1)
        concat_mask2 = torch.cat([out_up_mask2, mask1], dim=1)
        out_integral_conv2, out_integral_mask2 = self.integral_conv2(concat2, concat_mask2)

        out_up_conv1, out_up_mask1 = self.up_conv1(out_integral_conv2, out_integral_mask2)
        out_integral_conv1, out_integral_mask1 = self.integral_conv1(out_up_conv1, out_up_mask1)

        basis_depth, _ = self.predict_output(out_integral_conv1, out_integral_mask1)
        aggregated_depth = torch.mean(basis_depth, dim=2)
        normalized_depth = aggregated_depth / torch.mean(aggregated_depth, dim=[1, 2, 3])

        return normalized_depth
