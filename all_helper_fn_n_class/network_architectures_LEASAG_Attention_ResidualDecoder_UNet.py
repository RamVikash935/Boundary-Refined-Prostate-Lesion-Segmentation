#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ########


######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ########
######## Important: In compliance with the Apache License, we note that this file has     ########
########            been derived and modified from the original nnUNet work.              ########
######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ########


######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ########


import sys
import os

sys.path.append(os.path.abspath('/home/ss_students/zywx/main_file/all_helper_fn_n_class'))

import torch
import torch._dynamo

# Suppress Triton-related errors and fallback to eager mode
torch._dynamo.config.suppress_errors = True

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignores only UserWarnings

import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)  # Hides TorchDynamo warnings


import numpy as np
import torch

from network_architectures_neural_network import SegmentationNetwork

from copy import deepcopy
from torch import cat
from torch.nn.parameter import Parameter
import torch.jit
from torch.optim import SGD
from torch.backends import cudnn

import torch.nn as nn
import torch.nn.functional as F

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        out = nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                        align_corners=self.align_corners)
        return out


def get_default_network_config(dim=2, dropout_p=None, nonlin="LeakyReLU", norm_type="bn"):
    """
    returns a dictionary that contains pointers to conv, nonlin and norm ops and the default kwargs I like to use
    :return:
    """
    props = {}
    if dim == 2:
        props['conv_op'] = nn.Conv2d
        props['dropout_op'] = nn.Dropout2d
    elif dim == 3:
        props['conv_op'] = nn.Conv3d
        props['dropout_op'] = nn.Dropout3d
    else:
        raise NotImplementedError

    if norm_type == "bn":
        if dim == 2:
            props['norm_op'] = nn.BatchNorm2d
        elif dim == 3:
            props['norm_op'] = nn.BatchNorm3d
        props['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True}
    elif norm_type == "in":
        if dim == 2:
            props['norm_op'] = nn.InstanceNorm2d
        elif dim == 3:
            props['norm_op'] = nn.InstanceNorm3d
        props['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True}
    else:
        raise NotImplementedError

    if dropout_p is None:
        props['dropout_op'] = None
        props['dropout_op_kwargs'] = {'p': 0, 'inplace': True}
    else:
        props['dropout_op_kwargs'] = {'p': dropout_p, 'inplace': True}

    props['conv_op_kwargs'] = {'stride': 1, 'dilation': 1, 'bias': True}  # kernel size will be set by network!

    if nonlin == "LeakyReLU":
        props['nonlin'] = nn.LeakyReLU
        props['nonlin_kwargs'] = {'negative_slope': 1e-2, 'inplace': True}
    elif nonlin == "ReLU":
        props['nonlin'] = nn.ReLU
        props['nonlin_kwargs'] = {'inplace': True}
    else:
        raise ValueError

    if dim == 2:
        props['avg_pool'] = nn.AvgPool2d
    elif dim == 3:
        props['avg_pool'] = nn.AvgPool3d

    return props


class ConvDropoutNormReLU(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props):
        """
        if network_props['dropout_op'] is None then no dropout
        if network_props['norm_op'] is None then no norm
        :param input_channels:
        :param output_channels:
        :param kernel_size:
        :param network_props:
        """        
        super(ConvDropoutNormReLU, self).__init__()
        network_props = deepcopy(network_props)
        self.conv = network_props['conv_op'](
            input_channels,
            output_channels,
            kernel_size,
            padding=[(i - 1) // 2 for i in kernel_size],
            **network_props['conv_op_kwargs']
        )
        # maybe dropout
        if network_props['dropout_op'] is not None:
            self.do = network_props['dropout_op'](**network_props['dropout_op_kwargs'])
        else:
            self.do = nn.Identity()
        if network_props['norm_op'] is not None:
            self.norm = network_props['norm_op'](output_channels, **network_props['norm_op_kwargs'])
        else:
            self.norm = nn.Identity()
        self.nonlin = network_props['nonlin'](**network_props['nonlin_kwargs'])

        self.all = nn.Sequential(self.conv, self.do, self.norm, self.nonlin)

    def forward(self, x):
        out = self.all(x)
        return out


class StackedConvLayers(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_convs, first_stride=None):
        """
        if network_props['dropout_op'] is None then no dropout
        if network_props['norm_op'] is None then no norm
        :param input_channels:
        :param output_channels:
        :param kernel_size:
        :param network_props:
        """
        super(StackedConvLayers, self).__init__()

        network_props = deepcopy(network_props)
        network_props_first = deepcopy(network_props)
    
        if first_stride is not None:
            network_props_first['conv_op_kwargs']['stride'] = first_stride

        layers = []
        first_layer = ConvDropoutNormReLU(input_channels, output_channels, kernel_size, network_props_first)
        layers.append(first_layer)

        for idx in range(num_convs - 1):
            layer = ConvDropoutNormReLU(output_channels, output_channels, kernel_size, network_props)
            layers.append(layer)
        self.convs = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.convs(x)
        return out

class BasicResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
        """
        This is the conv bn nonlin conv bn nonlin kind of block
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        """       
        super(BasicResidualBlock, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.props = deepcopy(props)

        self.props['conv_op_kwargs']['stride'] = 1

        if stride is not None:
            kwargs_conv1 = deepcopy(self.props['conv_op_kwargs'])
            kwargs_conv1['stride'] = stride
        else:
            kwargs_conv1 = self.props['conv_op_kwargs']

        self.conv1 = self.props['conv_op'](in_planes, out_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size], **kwargs_conv1)
        self.norm1 = self.props['norm_op'](out_planes, **self.props['norm_op_kwargs'])
        self.nonlin1 = self.props['nonlin'](**self.props['nonlin_kwargs'])
        
        if self.props.get('dropout_op_kwargs', {}).get('p', 0) != 0:
            self.dropout = self.props['dropout_op'](**self.props['dropout_op_kwargs'])
        else:
            self.dropout = nn.Identity()
            
        self.conv2 = self.props['conv_op'](out_planes, out_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size], **self.props['conv_op_kwargs'])
        self.norm2 = self.props['norm_op'](out_planes, **self.props['norm_op_kwargs'])
        self.nonlin2 = self.props['nonlin'](**self.props['nonlin_kwargs'])

        if (stride is not None and any(i != 1 for i in (stride if isinstance(stride, (list, tuple)) else [stride]))) or (in_planes != out_planes):
            stride_val = stride if stride is not None else 1
            
            self.downsample_skip = nn.Sequential(
                self.props['conv_op'](in_planes, out_planes, 1, stride_val, bias=False),
                self.props['norm_op'](out_planes, **self.props['norm_op_kwargs'])
            )
        
        else:
            self.downsample_skip = lambda x: x
            
    def forward(self, x):
        residual = x
        out = self.dropout(self.conv1(x))
        out = self.nonlin1(self.norm1(out))
        out = self.norm2(self.conv2(out))
        residual = self.downsample_skip(residual)
        out += residual
        out = self.nonlin2(out)
        return out



class ResidualLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_blocks, first_stride=None, block=BasicResidualBlock):
        super().__init__()
        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.convs = nn.Sequential(
            block(input_channels, output_channels, kernel_size, network_props, first_stride),
        )
        
        for i in range(num_blocks - 1):
            blk = block(output_channels, output_channels, kernel_size, network_props)
            self.convs.add_module(f"block_{i+1}", blk)

    def forward(self, x):
        for idx, layer in enumerate(self.convs):
            x = layer(x)
        return x


class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_planes, out_planes, props, kernel_size=(1,1,1), stride=(1,1,1)):
        super(UnetGridGatingSignal3, self).__init__()

        # Store parameters
        self.kernel_size = kernel_size
        self.stride = stride
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes

        # Update conv_op_kwargs
        props['conv_op_kwargs']['stride'] = stride
        # Build conv1 sequence
        self.conv1 = nn.Sequential(
            props['conv_op'](
                in_planes,
                out_planes,
                kernel_size,
                padding=(0, 0, 0),
                **props['conv_op_kwargs']
            ),
            props['norm_op'](out_planes, **props['norm_op_kwargs']),
            props['nonlin'](**props['nonlin_kwargs'])
        )

    def forward(self, x):
        out = self.conv1(x)
        return out




class GridAttentionBlock3D(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, props=None, theta_kernel=(3, 3, 3),
                 theta_stride=(2, 2, 2)):
        super(GridAttentionBlock3D, self).__init__()

        props['conv_op_kwargs']['stride'] = print(f"Set props['conv_op_kwargs']['stride'] to 1") or 1
        self.props = props

        if theta_stride is not None:
            kwargs_conv1 = deepcopy(props['conv_op_kwargs'])
            kwargs_conv1['stride'] = theta_stride
        else:
            kwargs_conv1 = props['conv_op_kwargs']

        # Downsampling rate for the input featuremap # will be stride and kernel for conv applied on x
        self.theta_kernel = theta_kernel
        self.theta_stride = theta_stride

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

###################################################################################################################
        # these are the operation executed on the gating and input ;
        # # note : sub_sample_kernel_size and sub_sample_factor
        # this is beacause, gating signal comes from deeper level, hence smaller feature map,
        # aim here is to make the feature map sizes of the input (i.e. skip connection) and gating signal the same
        # by default assume that gating signal feature map is half the size of the input feature map size.
        # this will need to be modified for nnunet
###################################################################################################################
        
        self.theta = self.props['conv_op'](
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=self.theta_kernel,
            padding=[(i - 1) // 2 for i in self.theta_kernel],
            **kwargs_conv1
        )

        # aim of theta is to get the input to ssme sie as giting signal which is feature map one level below,
        #hence use the dame operation from before

        # phi applied on gating g
        self.phi = self.props['conv_op'](
            in_channels=self.gating_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            padding=0,
            **props['conv_op_kwargs']
        )
        # corresponds to weights of the input, that need to be brought to meaningful weights
        self.psi = self.props['conv_op'](
            in_channels=self.inter_channels,
            out_channels=1,
            kernel_size=1,
            padding=0,
            **props['conv_op_kwargs']
        )
        
        # Output transform
        self.W = nn.Sequential(
            self.props['conv_op'](
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                padding=0,
                **props['conv_op_kwargs']
            ),
            self.props['norm_op'](self.in_channels, **props['norm_op_kwargs'])
        )
        
    def forward(self, x, g):
        output = self._concatenation(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0), "Batch size of x and g must match"

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.interpolate(
            self.phi(g),
            size=theta_x_size[2:],
            mode='trilinear',
            align_corners=False
        )
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = torch.sigmoid(self.psi(f))
        
        sigm_psi_f = F.interpolate(
            sigm_psi_f,
            size=input_size[2:],
            mode='trilinear',
            align_corners=False
        )
    
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y, sigm_psi_f
    

class EdgeAwareMSAG(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels, props, theta_kernel, theta_stride):
        super(EdgeAwareMSAG, self).__init__()
        self.grid_attention = GridAttentionBlock3D(
            in_channels=in_channels,
            gating_channels=gating_channels,
            inter_channels=inter_channels,
            props=props,
            theta_kernel=theta_kernel,
            theta_stride=theta_stride
        )
        # Edge pathway: simple Laplacian filter for 3D edge enhancement
        self.edge_conv = nn.Conv3d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        
        laplacian_kernel = torch.tensor(
            [[[[[0, 0, 0], [0, -1, 0], [0, 0, 0]],
                [[0, -1, 0], [-1, 6, -1], [0, -1, 0]],
                [[0, 0, 0], [0, -1, 0], [0, 0, 0]]]]], dtype=torch.float32
        )
        with torch.no_grad():
            self.edge_conv.weight.copy_(laplacian_kernel)
        self.edge_conv.weight.requires_grad = False

        # Combine attention and edge maps
        self.combine = nn.Sequential(
            nn.Conv3d(in_channels + 1, in_channels, kernel_size=1),
            props['norm_op'](in_channels, **props['norm_op_kwargs']),
            props['nonlin'](**props['nonlin_kwargs'])
        )
    def forward(self, x, g):
        gated_x, attention_map = self.grid_attention(x, g)
        edge_map = self.edge_conv(x)
        fused = torch.cat([gated_x, edge_map], dim=1)
        out = self.combine(fused)
        return out, attention_map  # <--- this line was missing the second return value


class LearnableEdgeAwareMSAG(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels, props, theta_kernel, theta_stride):
        super(LearnableEdgeAwareMSAG, self).__init__()
        self.grid_attention = GridAttentionBlock3D(
            in_channels=in_channels,
            gating_channels=gating_channels,
            inter_channels=inter_channels,
            props=props,
            theta_kernel=theta_kernel,
            theta_stride=theta_stride
        )
        self.edge_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=True),
            props['norm_op'](in_channels // 2, **props['norm_op_kwargs']),
            props['nonlin'](**props['nonlin_kwargs']),
            nn.Conv3d(in_channels // 2, 1, kernel_size=1, padding=0, bias=True)
        )
        # Combine attention and edge maps
        self.combine = nn.Sequential(
            nn.Conv3d(in_channels + 1, in_channels, kernel_size=1),
            props['norm_op'](in_channels, **props['norm_op_kwargs']),
            props['nonlin'](**props['nonlin_kwargs'])
        )
    def forward(self, x, g):
        gated_x, attention_map = self.grid_attention(x, g)
        edge_map = self.edge_conv(x)
        fused = torch.cat([gated_x, edge_map], dim=1)
        out = self.combine(fused)
        return out, attention_map  
    


class SingleAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, props, theta_kernel,
                 theta_stride):
        super(SingleAttentionBlock, self).__init__()
        
        self.gate_block_1 = GridAttentionBlock3D(
            in_channels=in_size,
            gating_channels=gate_size,
            inter_channels=inter_size,
            props=props,
            theta_kernel=theta_kernel,
            theta_stride=theta_stride
        )
    def forward(self, x, gating_signal):
        gate_1, attention_1 = self.gate_block_1(x, gating_signal)
        return gate_1, attention_1


class ResidualUNetDecoder(nn.Module):
    def __init__(self, previous, num_classes, num_blocks_per_stage=None, network_props=None, deep_supervision=False,
                 upscale_logits=False, block=BasicResidualBlock):
        
        super(ResidualUNetDecoder, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        # We assume the bottleneck is part of the encoder, so we can start with upsample -> concat here
        previous_stages = previous.stages
        previous_stage_output_features = previous.stage_output_features
        previous_stage_pool_kernel_size = previous.stage_pool_kernel_size
        previous_stage_conv_op_kernel_size = previous.stage_conv_op_kernel_size
        if network_props is None:
            self.props = previous.props
        else:
            self.props = network_props
        if self.props['conv_op'] == nn.Conv2d:
            transpconv = nn.ConvTranspose2d
            upsample_mode = "bilinear"
        elif self.props['conv_op'] == nn.Conv3d:
            transpconv = nn.ConvTranspose3d
            upsample_mode = "trilinear"
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(self.props['conv_op']))

        if num_blocks_per_stage is None:
            num_blocks_per_stage = previous.num_blocks_per_stage[:-1][::-1]
        assert len(num_blocks_per_stage) == len(previous.num_blocks_per_stage) - 1
        self.stage_pool_kernel_size = previous_stage_pool_kernel_size
        self.stage_output_features = previous_stage_output_features
        self.stage_conv_op_kernel_size = previous_stage_conv_op_kernel_size

        num_stages = len(previous_stages) - 1  # one less due to bottleneck
        self.tus = []
        self.stages = []
        self.deep_supervision_outputs = []
        # cumulate upsample factors for logit upsampling
        cum_upsample = np.cumprod(np.vstack(self.stage_pool_kernel_size), axis=0).astype(int)
        for i, s in enumerate(np.arange(num_stages)[::-1]):
            features_below = previous_stage_output_features[s + 1]
            features_skip = previous_stage_output_features[s]
            tu = transpconv(features_below, features_skip, previous_stage_pool_kernel_size[s + 1],
                           previous_stage_pool_kernel_size[s + 1], bias=False)
            self.tus.append(tu)
            # after we tu we concat features so now we have 2xfeatures_skip
            stage = ResidualLayer(
                2 * features_skip, features_skip,
                previous_stage_conv_op_kernel_size[s],
                self.props, num_blocks_per_stage[i], None, block
            )
            self.stages.append(stage)
            if deep_supervision and s != 0:
                seg_layer = self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, False)
                if upscale_logits:
                    upsample = nn.Upsample(scale_factor=cum_upsample[s], mode=upsample_mode)
                    self.deep_supervision_outputs.append(nn.Sequential(seg_layer, upsample))
                else:
                    self.deep_supervision_outputs.append(seg_layer)
        self.segmentation_output = self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, False)
        self.stages = nn.ModuleList(self.stages)
        self.deep_supervision_outputs = nn.ModuleList(self.deep_supervision_outputs)

    def forward(self, skips):
        # skips come from the encoder. They are sorted so that the bottleneck is last in the list
        # what is maybe not perfect is that the TUs and stages here are sorted the other way around
        # so let's just reverse the order of skips        
        skips = skips[::-1]
        seg_outputs = []
        x = skips[0]
        
        for i in range(len(self.tus)):
            x = self.tus[i](x)
            x = torch.cat((x, skips[i + 1]), dim=1)
            x = self.stages[i](x)
            if self.deep_supervision and (i != len(self.tus) - 1):
                out = self.deep_supervision_outputs[i](x)
                seg_outputs.append(out)

        segmentation = self.segmentation_output(x)
        if self.deep_supervision:
            seg_outputs.append(segmentation)
            result = seg_outputs[::-1]  # seg_outputs are ordered so that the seg from the highest layer is first, the seg from
            # the bottleneck of the UNet last
            return result
        else:
            return segmentation

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_classes, pool_op_kernel_sizes, num_blocks_per_stage_decoder,
                                        feat_map_mul_on_downscale, batch_size):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :return:
        """
        npool = len(pool_op_kernel_sizes) - 1

        current_shape = np.array(patch_size)
        tmp = (num_blocks_per_stage_decoder[-1] * 2 + 1) * np.prod(
            current_shape) * base_num_features + num_classes * np.prod(current_shape)

        num_feat = base_num_features

        for p in range(1, npool):
            current_shape = current_shape / np.array(pool_op_kernel_sizes[p])
            num_feat = min(num_feat * feat_map_mul_on_downscale, max_num_features)
            num_convs = num_blocks_per_stage_decoder[-(p + 1)] * 2 + 1 + 1  # +1 for transpconv and +1 for conv in skip
            print(p, num_feat, num_convs, current_shape)
            tmp += num_convs * np.prod(current_shape) * num_feat

        return tmp * batch_size



    
class LearnableEASAGAttentionResidualDecoder(nn.Module):
    def __init__(self, previous, num_classes, num_blocks_per_stage=None, network_props=None, deep_supervision=False,
                 upscale_logits=False, block=BasicResidualBlock):
        
        super(LearnableEASAGAttentionResidualDecoder, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        # We assume the bottleneck is part of the encoder, so we can start with upsample -> concat here
        previous_stages = previous.stages
        previous_stage_output_features = previous.stage_output_features
        previous_stage_pool_kernel_size = previous.stage_pool_kernel_size
        previous_stage_conv_op_kernel_size = previous.stage_conv_op_kernel_size
        if network_props is None:
            self.props = previous.props
        else:
            self.props = network_props
            
        if self.props['conv_op'] == nn.Conv2d:
            transpconv = nn.ConvTranspose2d
            upsample_mode = "bilinear"
        elif self.props['conv_op'] == nn.Conv3d:
            transpconv = nn.ConvTranspose3d
            upsample_mode = "trilinear"
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(self.props['conv_op']))

        if num_blocks_per_stage is None:
            num_blocks_per_stage = previous.num_blocks_per_stage[:-1][::-1]

        assert len(num_blocks_per_stage) == len(previous.num_blocks_per_stage) - 1
    
        self.stage_pool_kernel_size = previous_stage_pool_kernel_size
        self.stage_output_features = previous_stage_output_features
        self.stage_conv_op_kernel_size = previous_stage_conv_op_kernel_size

        num_stages = len(previous_stages) - 1  # one less due to bottleneck

        self.tus = []
        self.atten = []  # contains attention operations ############
        self.stages = []
        self.deep_supervision_outputs = []
        
        self.gating = UnetGridGatingSignal3(previous_stage_output_features[-1], previous_stage_output_features[-1],
                                            props=self.props, kernel_size=(1,1,1), stride=(1,1,1))  # meant to correspond to bn channels/features
        # only used for upsample_logits
        cum_upsample = np.cumprod(np.vstack(self.stage_pool_kernel_size), axis=0).astype(int)
        for i, s in enumerate(np.arange(num_stages)[::-1]):
            features_below = previous_stage_output_features[s + 1]
            features_skip = previous_stage_output_features[s]
            skip_conv_kernel_size = previous_stage_conv_op_kernel_size[s]
            skip_pool_kernel_size = previous_stage_pool_kernel_size[s]
        
            tu = transpconv(features_below, features_skip, previous_stage_pool_kernel_size[s + 1],
                            previous_stage_pool_kernel_size[s + 1], bias=False)
            self.tus.append(tu)
            
            att_block = LearnableEdgeAwareMSAG(in_channels=features_skip, gating_channels=features_below, inter_channels=features_skip,
                                    props=self.props, theta_kernel=skip_conv_kernel_size,
                                    theta_stride=skip_pool_kernel_size)
            
            self.atten.append(att_block)

            # after we tu we concat features so now we have 2xfeatures_skip
            stage_block = ResidualLayer(2 * features_skip, features_skip,
                                            previous_stage_conv_op_kernel_size[s], self.props,
                                            num_blocks_per_stage[i], None, block)
            self.stages.append(stage_block)

            if deep_supervision and s != 0:
                seg_layer = self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, False)
                print("seg_layer =", seg_layer)
                if upscale_logits:
                    upsample = Upsample(scale_factor=cum_upsample[s], mode=upsample_mode)
                    self.deep_supervision_outputs.append(nn.Sequential(seg_layer, upsample))
                else:
                    self.deep_supervision_outputs.append(seg_layer)

        self.segmentation_output = self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, False)
        self.tus = nn.ModuleList(self.tus)
        self.atten = nn.ModuleList(self.atten)
        self.stages = nn.ModuleList(self.stages)
        self.deep_supervision_outputs = nn.ModuleList(self.deep_supervision_outputs)

    def forward(self, skips):
        # skips come from the encoder. They are sorted so that the bottleneck is last in the list
        # what is maybe not perfect is that the TUs and stages here are sorted the other way around
        # so let's just reverse the order of skips        
        skips = skips[::-1]
        seg_outputs = []
        x = skips[0]     # this is the bottleneck
        g = self.gating(x)
        for i in range(len(self.tus)):
            skip_input = skips[i + 1] # 0 is bottleneck
            g_conv, att = self.atten[i](skip_input, g)
            x = self.tus[i](x)
            x = torch.cat((x, g_conv), dim=1)
            x = self.stages[i](x)  # apply the conv layers post upsampling
            g = x

            if self.deep_supervision and (i != len(self.tus) - 1):
                output = self.deep_supervision_outputs[i](x)
                seg_outputs.append(output)

        segmentation = self.segmentation_output(x)
        if self.deep_supervision:
            seg_outputs.append(segmentation)
            return seg_outputs[::-1] # seg_outputs are ordered so that the seg from the highest layer is first, the seg from
            # the bottleneck of the UNet last
        else:
            return segmentation

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_classes, pool_op_kernel_sizes, num_blocks_per_stage_decoder,
                                        feat_map_mul_on_downscale, batch_size):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :return:
        """
        npool = len(pool_op_kernel_sizes) - 1

        current_shape = np.array(patch_size)
        tmp = (num_blocks_per_stage_decoder[-1] * 2 + 1) * np.prod(
            current_shape) * base_num_features + num_classes * np.prod(current_shape)

        num_feat = base_num_features

        for p in range(1, npool):
            current_shape = current_shape / np.array(pool_op_kernel_sizes[p])
            num_feat = min(num_feat * feat_map_mul_on_downscale, max_num_features)
            num_convs = num_blocks_per_stage_decoder[-(p + 1)] * 2 + 1 + 1  # +1 for transpconv and +1 for conv in skip
            print(p, num_feat, num_convs, current_shape)
            tmp += num_convs * np.prod(current_shape) * num_feat

        return tmp * batch_size


class GenericAttentionUNetEncoder(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, default_return_skips=True,
                 max_num_features=320):
        """
        Following UNet building blocks can be added by utilizing the properties this class exposes (TODO)

        this one includes the bottleneck layer!

        :param input_channels:
        :param base_num_features:
        :param num_blocks_per_stage:
        :param feat_map_mul_on_downscale:
        :param pool_op_kernel_sizes:
        :param conv_kernel_sizes:
        :param props:
        """
        super(GenericAttentionUNetEncoder, self).__init__()

        self.default_return_skips = default_return_skips
        self.props = props

        self.stages = []
        self.stage_output_features = []
        self.stage_pool_kernel_size = []
        self.stage_conv_op_kernel_size = []

        assert len(pool_op_kernel_sizes) == len(conv_kernel_sizes)

        num_stages = len(conv_kernel_sizes)
    
        if not isinstance(num_blocks_per_stage, (list, tuple)):
            num_blocks_per_stage = [num_blocks_per_stage] * num_stages
        else:
            assert len(num_blocks_per_stage) == num_stages

        self.num_blocks_per_stage = num_blocks_per_stage  # decoder may need this
        current_input_features = input_channels
        
        for stage in range(num_stages):
            current_output_features = min(base_num_features * feat_map_mul_on_downscale ** stage, max_num_features)
            current_kernel_size = conv_kernel_sizes[stage]
            current_pool_kernel_size = pool_op_kernel_sizes[stage]
            current_stage = StackedConvLayers(current_input_features, current_output_features, current_kernel_size,
                                              props, self.num_blocks_per_stage[stage], current_pool_kernel_size)
            
            self.stages.append(current_stage)
            self.stage_output_features.append(current_output_features)
            self.stage_conv_op_kernel_size.append(current_kernel_size)
            self.stage_pool_kernel_size.append(current_pool_kernel_size)
 
            # update current_input_features
            current_input_features = current_output_features

        self.stages = nn.ModuleList(self.stages)

    def forward(self, x, return_skips=None):
        """
        :param x:
        :param return_skips: if none then self.default_return_skips is used
        :return:
        """
        skips = []

        for idx, s in enumerate(self.stages):
            x = s(x)
            if self.default_return_skips:
                skips.append(x)

        if return_skips is None:
            return_skips = self.default_return_skips

        if return_skips:
            return skips
        else:
            return x

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, pool_op_kernel_sizes, num_conv_per_stage_encoder,
                                        feat_map_mul_on_downscale, batch_size):
        npool = len(pool_op_kernel_sizes) - 1

        current_shape = np.array(patch_size)

        tmp = (num_conv_per_stage_encoder[0] * 2 + 1) * np.prod(current_shape) * base_num_features \
              + num_modalities * np.prod(current_shape)

        num_feat = base_num_features

        for p in range(1, npool + 1):
            current_shape = current_shape / np.array(pool_op_kernel_sizes[p])
            num_feat = min(num_feat * feat_map_mul_on_downscale, max_num_features)
            num_convs = num_conv_per_stage_encoder[p] * 2 + 1  # + 1 for conv in skip in first block
            print(p, num_feat, num_convs, current_shape)
            tmp += num_convs * np.prod(current_shape) * num_feat
        return tmp * batch_size


class LearnableEASAGResidualDecoderAttentionUNet(SegmentationNetwork):

    use_this_for_2D_configuration = 622116860.5  # 1167982592.0
    use_this_for_3D_configuration = 622116860.5

    use_this_for_batch_size_computation_2D = 858931200.0  # 1167982592.0
    use_this_for_batch_size_computation_3D = 727842816.0  # 1152286720.0
    default_base_num_features = 32
    default_conv_per_stage = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    default_blocks_per_stage_encoder = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    default_blocks_per_stage_decoder = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    default_min_batch_size = 2  # this is what works with the numbers above

    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=320, initializer=None,
                 block=BasicResidualBlock):
        super(LearnableEASAGResidualDecoderAttentionUNet, self).__init__()
        self.conv_op = props['conv_op']
        self.num_classes = num_classes

        self.encoder = GenericAttentionUNetEncoder(input_channels, base_num_features, num_blocks_per_stage_encoder,
                                                    feat_map_mul_on_downscale, pool_op_kernel_sizes, conv_kernel_sizes,
                                                    props, default_return_skips=True, max_num_features=max_features)
        
        self.decoder = LearnableEASAGAttentionResidualDecoder(self.encoder, num_classes, num_blocks_per_stage_decoder, props,
                                                    deep_supervision, upscale_logits, block=block)
        
        if initializer is not None:
            self.apply(initializer)

    def forward(self, x):
        skips = self.encoder(x)
        output = self.decoder(skips)
        return output

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, num_conv_per_stage_encoder,
                                        num_conv_per_stage_decoder, feat_map_mul_on_downscale, batch_size):
        enc = GenericAttentionUNetEncoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                  num_modalities, pool_op_kernel_sizes,
                                                                  num_conv_per_stage_encoder,
                                                                  feat_map_mul_on_downscale, batch_size)
        dec = ResidualUNetDecoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                  num_classes, pool_op_kernel_sizes,
                                                                  num_conv_per_stage_decoder,
                                                                  feat_map_mul_on_downscale, batch_size)
        return enc + dec    