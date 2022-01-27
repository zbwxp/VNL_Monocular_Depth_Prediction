import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.cnn.bricks import build_norm_layer

# from .aspp_head import ASPPHead, ASPPModule
from .mix_transformer import Block


# from ..builder import HEADS


class DynHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 norm_cfg,
                 act_cfg,
                 upsample_f,
                 dyn_ch,
                 mask_ch,
                 use_low_level_info=False,
                 channel_reduce_factor=1):
        super(DynHead, self).__init__()

        channels = dyn_ch
        num_bases = 0
        if use_low_level_info:
            num_bases = mask_ch
        num_out_channel = (2 + num_bases) * channels + \
                          channels + \
                          channels * channels + \
                          channels + \
                          channels * num_classes + \
                          num_classes

        self.classifier = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels // channel_reduce_factor,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg, ),
            nn.Conv2d(in_channels // channel_reduce_factor, num_out_channel, 1)
        )

        nn.init.xavier_normal_(self.classifier[-1].weight)
        param = self.classifier[-1].weight / num_out_channel
        self.classifier[-1].weight = nn.Parameter(param)
        nn.init.constant_(self.classifier[-1].bias, 0)

    def forward(self, feature):
        return self.classifier(feature)


# @HEADS.register_module()
class nrd_trans_stage5(nn.Module):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, in_channels=512,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 num_classes=150,
                 c1_in_channels=64,
                 c1_channels=48,
                 upsample_factor=8,
                 dyn_branch_ch=16,
                 mask_head_ch=16,
                 feature_strides=None,
                 channel_reduce_factor=1,
                 encoder_stages=6,
                 **kwargs):
        super(nrd_trans_stage5, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        self.act_cfg = dict(type='ReLU')
        self.conv_cfg = None
        norm_layer = nn.LayerNorm
        self.block5 = nn.ModuleList([Block(
            dim=self.in_channels, num_heads=8, mlp_ratio=4, qkv_bias=True, qk_scale=None,
            drop=0.0, attn_drop=0.0, drop_path=0.1, norm_layer=norm_layer,
            sr_ratio=1)
            for i in range(encoder_stages)])
        self.norm4 = norm_layer(self.in_channels)
        for blk in self.block5:
            blk.apply(blk._init_weights)

        self.pad_out_channel = self.num_classes
        self.upsample_f = upsample_factor
        self.dyn_ch = dyn_branch_ch
        self.mask_ch = mask_head_ch
        self.use_low_level_info = True

        last_stage_ch = self.in_channels
        self.classifier = DynHead(last_stage_ch,
                                  self.pad_out_channel,
                                  self.norm_cfg,
                                  self.act_cfg,
                                  self.upsample_f,
                                  self.dyn_ch,
                                  self.mask_ch,
                                  self.use_low_level_info,
                                  channel_reduce_factor)

        if c1_in_channels > 0:
            self.c1_bottleneck = nn.Sequential(
                ConvModule(
                    c1_in_channels,
                    c1_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    c1_channels,
                    self.mask_ch,
                    1,
                    conv_cfg=self.conv_cfg,
                    act_cfg=None,
                ),
            )
        else:
            self.c1_bottleneck = None

        _, norm = build_norm_layer(self.norm_cfg, 2 + self.mask_ch)
        self.add_module("cat_norm", norm)
        nn.init.constant_(self.cat_norm.weight, 1)
        nn.init.constant_(self.cat_norm.bias, 0)
        self.smooth = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.num_classes,
                self.num_classes,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.num_classes,
                self.num_classes,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        )


    def forward(self, inputs):
        """Forward function."""
        x = inputs[-1]
        B, c, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block5):
            x = blk(x, H, W)
        x = self.norm4(x)
        output = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])

        output = self.classifier(output)
        output = self.interpolate_fast(output, c1_output, self.cat_norm)
        output = self.smooth(output)
        return output

    def interpolate_fast(self, x, x_cat=None, norm=None):
        dy_ch = self.dyn_ch
        B, conv_ch, H, W = x.size()
        weights, biases = self.get_subnetworks_params_fast(x, channels=dy_ch)
        f = self.upsample_f
        self.coord_generator(H, W)
        coord = self.coord.reshape(1, H, W, 2, f, f).permute(0, 3, 1, 4, 2, 5).reshape(1, 2, H * f, W * f)
        coord = coord.repeat(B, 1, 1, 1)
        if x_cat is not None:
            coord = torch.cat((coord, x_cat), 1)
            coord = norm(coord)

        output = self.subnetworks_forward_fast(coord, weights, biases, B * H * W)
        return output

    def get_subnetworks_params_fast(self, attns, num_bases=0, channels=16):
        assert attns.dim() == 4
        B, conv_ch, H, W = attns.size()
        if self.use_low_level_info:
            num_bases = self.mask_ch
        else:
            num_bases = 0

        w0, b0, w1, b1, w2, b2 = torch.split_with_sizes(attns, [
            (2 + num_bases) * channels, channels,
            channels * channels, channels,
            channels * self.pad_out_channel, self.pad_out_channel
        ], dim=1)

        # out_channels x in_channels x 1 x 1
        w0 = F.interpolate(w0, scale_factor=self.upsample_f, mode='nearest')
        b0 = F.interpolate(b0, scale_factor=self.upsample_f, mode='nearest')
        w1 = F.interpolate(w1, scale_factor=self.upsample_f, mode='nearest')
        b1 = F.interpolate(b1, scale_factor=self.upsample_f, mode='nearest')
        w2 = F.interpolate(w2, scale_factor=self.upsample_f, mode='nearest')
        b2 = F.interpolate(b2, scale_factor=self.upsample_f, mode='nearest')

        return [w0, w1, w2], [b0, b1, b2]

    def subnetworks_forward_fast(self, inputs, weights, biases, n_subnets):
        assert inputs.dim() == 4
        n_layer = len(weights)
        x = inputs
        if self.use_low_level_info:
            num_bases = self.mask_ch
        else:
            num_bases = 0
        for i, (w, b) in enumerate(zip(weights, biases)):
            if i == 0:
                x = self.padconv(x, w, b, cin=2 + num_bases, cout=self.dyn_ch, relu=True)
            if i == 1:
                x = self.padconv(x, w, b, cin=self.dyn_ch, cout=self.dyn_ch, relu=True)
            if i == 2:
                x = self.padconv(x, w, b, cin=self.dyn_ch, cout=self.pad_out_channel, relu=False)
        return x

    def padconv(self, input, w, b, cin, cout, relu):
        input = input.repeat(1, cout, 1, 1)
        x = input * w
        conv_w = torch.ones((cout, cin, 1, 1), device=input.device)
        x = F.conv2d(
            x, conv_w, stride=1, padding=0,
            groups=cout
        )
        x = x + b
        if relu:
            x = F.relu(x)
        return x

    def coord_generator(self, height, width):
        f = self.upsample_f
        coord = compute_locations_per_level(f, f)
        H = height
        W = width
        coord = coord.repeat(H * W, 1, 1, 1)
        self.coord = coord.to(device='cuda')


def compute_locations_per_level(h, w):
    shifts_x = torch.arange(
        0, 1, step=1 / w,
        dtype=torch.float32, device='cuda'
    )
    shifts_y = torch.arange(
        0, 1, step=1 / h,
        dtype=torch.float32, device='cuda'
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    locations = torch.stack((shift_x, shift_y), dim=0)
    return locations
