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

from copy import deepcopy
from utilities import softmax_helper
from torch import nn
import torch
import numpy as np
from network_architectures_initialization import InitWeights_He
from network_architectures_neural_network import SegmentationNetwork
import torch.nn.functional


class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        
        super(ConvDropoutNormNonlin, self).__init__()
        print(f"[DEBUG] Initializing ConvDropoutNormNonlin with in_ch={input_channels}, out_ch={output_channels}, nonlin={nonlin.__name__}")
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
            print(f"[DEBUG] nonlin_kwargs defaulted: {nonlin_kwargs}")
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
            print(f"[DEBUG] dropout_op_kwargs defaulted: {dropout_op_kwargs}")
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
            print(f"[DEBUG] norm_op_kwargs defaulted: {norm_op_kwargs}")
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
            print(f"[DEBUG] conv_kwargs defaulted: {conv_kwargs}")

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        print(f"[DEBUG] conv layer created: {self.conv}")
        
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs['p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
            print(f"[DEBUG] dropout layer created: {self.dropout}")
        else:
            self.dropout = None
            print(f"[DEBUG] dropout disabled")
        
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        print(f"[DEBUG] norm layer created: {self.instnorm}")
        
        self.lrelu = self.nonlin(**self.nonlin_kwargs)
        print(f"[DEBUG] nonlin layer created: {self.lrelu}")

    def forward(self, x):
        # print(f"[DEBUG] forward input shape: {x.shape}")
        x = self.conv(x)
        # print(f"[DEBUG] after conv shape: {x.shape}")
        if self.dropout is not None:
            x = self.dropout(x)
            # print(f"[DEBUG] after dropout shape: {x.shape}")
        x = self.instnorm(x)
        # print(f"[DEBUG] after norm shape: {x.shape}")
        x = self.lrelu(x)
        # print(f"[DEBUG] after nonlin shape: {x.shape}")
        return x


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        # print(f"[DEBUG] ConvDropoutNonlinNorm forward input shape: {x.shape}")
        x = self.conv(x)
        # print(f"[DEBUG] after conv shape: {x.shape}")
        if self.dropout is not None:
            x = self.dropout(x)
            # print(f"[DEBUG] after dropout shape: {x.shape}")
        x = self.lrelu(x)
        # print(f"[DEBUG] after nonlin shape: {x.shape}")
        x = self.instnorm(x)
        # print(f"[DEBUG] after norm shape: {x.shape}")
        return x



class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        print(f"[DEBUG] Initializing StackedConvLayers with in_ch={input_feature_channels}, out_ch={output_feature_channels}, num_convs={num_convs}")
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
            print(f"[DEBUG] nonlin_kwargs defaulted: {nonlin_kwargs}")
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
            print(f"[DEBUG] dropout_op_kwargs defaulted: {dropout_op_kwargs}")
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
            print(f"[DEBUG] norm_op_kwargs defaulted: {norm_op_kwargs}")
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
            print(f"[DEBUG] conv_kwargs defaulted: {conv_kwargs}")
        
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
            print(f"[DEBUG] conv_kwargs_first_conv set with stride {first_stride}: {self.conv_kwargs_first_conv}")
        else:
            self.conv_kwargs_first_conv = conv_kwargs
            print(f"[DEBUG] conv_kwargs_first_conv defaulted: {self.conv_kwargs_first_conv}")

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))
        print(f"[DEBUG] StackedConvLayers blocks created with {len(self.blocks)} layers")

    def forward(self, x):
        # print(f"[DEBUG] StackedConvLayers forward input shape: {x.shape}")
        out = self.blocks(x)
        # print(f"[DEBUG] StackedConvLayers forward output shape: {out.shape}")
        return out


def print_module_training_status(module):
    print(f"[DEBUG] checking module training status for: {module}")
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(f"[DEBUG] Module: {module}, training: {module.training}")


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        print(f"Initializing Upsample with size={size}, scale_factor={scale_factor}, mode={mode}, align_corners={align_corners}")
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        print(f"Set align_corners: {self.align_corners}")
        self.mode = mode
        print(f"Set mode: {self.mode}")
        self.scale_factor = scale_factor
        print(f"Set scale_factor: {self.scale_factor}")
        self.size = size
        print(f"Set size: {self.size}")

    def forward(self, x):
        # print(f"Upsample forward input shape: {x.shape}")
        out = nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                        align_corners=self.align_corners)
        # print(f"Upsample output shape: {out.shape}")
        return out



class Generic_UNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440
        
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False,
                 convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """        
        super(Generic_UNet, self).__init__()
        print("Debug: Called Generic_UNet.__init__ and super init")
        self.convolutional_upsampling = convolutional_upsampling
        print(f"Debug: convolutional_upsampling set to {self.convolutional_upsampling}")
        self.convolutional_pooling = convolutional_pooling
        print(f"Debug: convolutional_pooling set to {self.convolutional_pooling}")
        self.upscale_logits = upscale_logits
        print(f"Debug: upscale_logits set to {self.upscale_logits}")
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        print(f"Debug: nonlin_kwargs resolved to {nonlin_kwargs}")
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        print(f"Debug: dropout_op_kwargs resolved to {dropout_op_kwargs}")
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        print(f"Debug: norm_op_kwargs resolved to {norm_op_kwargs}")

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        print(f"Debug: conv_kwargs initialized to {self.conv_kwargs}")

        self.nonlin = nonlin
        print(f"Debug: nonlin set to {self.nonlin}")
        self.nonlin_kwargs = nonlin_kwargs
        print(f"Debug: nonlin_kwargs attribute set")
        self.dropout_op_kwargs = dropout_op_kwargs
        print(f"Debug: dropout_op_kwargs attribute set")
        self.norm_op_kwargs = norm_op_kwargs
        print(f"Debug: norm_op_kwargs attribute set")
        self.weightInitializer = weightInitializer
        print(f"Debug: weightInitializer set")
        self.conv_op = conv_op
        print(f"Debug: conv_op set to {self.conv_op}")
        self.norm_op = norm_op
        print(f"Debug: norm_op set to {self.norm_op}")
        self.dropout_op = dropout_op
        print(f"Debug: dropout_op set to {self.dropout_op}")
        self.num_classes = num_classes
        print(f"Debug: num_classes set to {self.num_classes}")
        self.final_nonlin = final_nonlin
        print(f"Debug: final_nonlin set")
        self._deep_supervision = deep_supervision
        print(f"Debug: deep_supervision set to {self._deep_supervision}")
        self.do_ds = deep_supervision
        print(f"Debug: do_ds set to {self.do_ds}")

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            print(f"Debug: upsample_mode set to {upsample_mode}")
            pool_op = nn.MaxPool2d
            print(f"Debug: pool_op set to {pool_op}")
            transpconv = nn.ConvTranspose2d
            print(f"Debug: transpconv set to {transpconv}")
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            print(f"Debug: pool_op_kernel_sizes resolved to {pool_op_kernel_sizes}")
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
            print(f"Debug: conv_kernel_sizes resolved to {conv_kernel_sizes}")
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            print(f"Debug: upsample_mode set to {upsample_mode}")
            pool_op = nn.MaxPool3d
            print(f"Debug: pool_op set to {pool_op}")
            transpconv = nn.ConvTranspose3d
            print(f"Debug: transpconv set to {transpconv}")
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            print(f"Debug: pool_op_kernel_sizes resolved to {pool_op_kernel_sizes}")
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
            print(f"Debug: conv_kernel_sizes resolved to {conv_kernel_sizes}")
        else:
            raise ValueError(f"unknown convolution dimensionality, conv op: {str(conv_op)}")
        print("Debug: Convolution dimensionality check complete")

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        print(f"Debug: input_shape_must_be_divisible_by set to {self.input_shape_must_be_divisible_by}")
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        print("Debug: pool_op_kernel_sizes attribute set")
        self.conv_kernel_sizes = conv_kernel_sizes
        print("Debug: conv_kernel_sizes attribute set")

        self.conv_pad_sizes = []
        print("Debug: Initialized conv_pad_sizes list")
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])
            print(f"Debug: Added pad sizes {[1 if i == 3 else 0 for i in krnl]} for kernel {krnl}")

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features
        print(f"Debug: max_num_features set to {self.max_num_features}")

        self.conv_blocks_context = []
        print("Debug: Initialized conv_blocks_context list")
        self.conv_blocks_localization = []
        print("Debug: Initialized conv_blocks_localization list")
        self.td = []
        print("Debug: Initialized td list")
        self.tu = []
        print("Debug: Initialized tu list")
        self.seg_outputs = []
        print("Debug: Initialized seg_outputs list")

        output_features = base_num_features
        print(f"Debug: output_features set to {output_features}")
        input_features = input_channels
        print(f"Debug: input_features set to {input_features}")

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None
            print(f"Debug [context loop d={d}]: first_stride = {first_stride}")

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            print(f"Debug [context loop d={d}]: conv_kwargs['kernel_size'] = {self.conv_kwargs['kernel_size']}")
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            print(f"Debug [context loop d={d}]: conv_kwargs['padding'] = {self.conv_kwargs['padding']}")

            # add convolutions
            self.conv_blocks_context.append(
                StackedConvLayers(
                    input_features, output_features, num_conv_per_stage,
                    self.conv_op, self.conv_kwargs,
                    self.norm_op, self.norm_op_kwargs,
                    self.dropout_op, self.dropout_op_kwargs,
                    self.nonlin, self.nonlin_kwargs,
                    first_stride, basic_block=basic_block
                )
            )
            print(f"Debug [context loop d={d}]: appended StackedConvLayers(input={input_features}, output={output_features})")

            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
                print(f"Debug [context loop d={d}]: appended pooling op with kernel {pool_op_kernel_sizes[d]}")

            input_features = output_features
            print(f"Debug [context loop d={d}]: input_features updated to {input_features}")
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            print(f"Debug [context loop d={d}]: output_features scaled to {output_features}")

            output_features = min(output_features, self.max_num_features)
            print(f"Debug [context loop d={d}]: output_features clipped to {output_features}")

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None
        print(f"Debug [bottleneck]: first_stride = {first_stride}")

        # choose final feature count
        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels
        print(f"Debug [bottleneck]: final_num_features = {final_num_features}")

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        print(f"Debug [bottleneck]: conv_kwargs['kernel_size'] = {self.conv_kwargs['kernel_size']}")
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        print(f"Debug [bottleneck]: conv_kwargs['padding'] = {self.conv_kwargs['padding']}")

        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(
                input_features, output_features, num_conv_per_stage - 1,
                self.conv_op, self.conv_kwargs,
                self.norm_op, self.norm_op_kwargs,
                self.dropout_op, self.dropout_op_kwargs,
                self.nonlin, self.nonlin_kwargs,
                first_stride, basic_block=basic_block
            ),
            StackedConvLayers(
                output_features, final_num_features, 1,
                self.conv_op, self.conv_kwargs,
                self.norm_op, self.norm_op_kwargs,
                self.dropout_op, self.dropout_op_kwargs,
                self.nonlin, self.nonlin_kwargs,
                basic_block=basic_block
            )
        ))
        print("Debug [bottleneck]: appended bottleneck conv blocks")

        # if we don't want dropout in localization, zero it out
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0
            print(f"Debug [localization init]: dropout p changed from {old_dropout_p} to {self.dropout_op_kwargs['p']}")

        # localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels
            n_features_after_tu_and_concat = nfeatures_from_skip * 2
            print(f"Debug [loc loop u={u}]: from_down={nfeatures_from_down}, from_skip={nfeatures_from_skip}")

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip
            print(f"Debug [loc loop u={u}]: final_num_features = {final_num_features}")

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(
                    scale_factor=pool_op_kernel_sizes[-(u + 1)],
                    mode=upsample_mode
                ))
                print(f"Debug [loc loop u={u}]: appended Upsample with scale {pool_op_kernel_sizes[-(u + 1)]}")
            else:
                self.tu.append(transpconv(
                    nfeatures_from_down, nfeatures_from_skip,
                    pool_op_kernel_sizes[-(u + 1)],
                    pool_op_kernel_sizes[-(u + 1)],
                    bias=False
                ))
                print(f"Debug [loc loop u={u}]: appended TransposedConv from {nfeatures_from_down} to {nfeatures_from_skip}")

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[-(u + 1)]
            print(f"Debug [loc loop u={u}]: conv_kwargs['kernel_size'] = {self.conv_kwargs['kernel_size']}")
            self.conv_kwargs['padding'] = self.conv_pad_sizes[-(u + 1)]
            print(f"Debug [loc loop u={u}]: conv_kwargs['padding'] = {self.conv_kwargs['padding']}")

            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(
                    n_features_after_tu_and_concat, nfeatures_from_skip,
                    num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                    self.norm_op, self.norm_op_kwargs,
                    self.dropout_op, self.dropout_op_kwargs,
                    self.nonlin, self.nonlin_kwargs,
                    basic_block=basic_block
                ),
                StackedConvLayers(
                    nfeatures_from_skip, final_num_features,
                    1, self.conv_op, self.conv_kwargs,
                    self.norm_op, self.norm_op_kwargs,
                    self.dropout_op, self.dropout_op_kwargs,
                    self.nonlin, self.nonlin_kwargs,
                    basic_block=basic_block
                )
            ))
            print(f"Debug [loc loop u={u}]: appended localization conv blocks")

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(
                conv_op(
                    self.conv_blocks_localization[ds][-1].output_channels,
                    num_classes, 1, 1, 0, 1, 1, seg_output_use_bias
                )
            )
            print(f"Debug [seg head ds={ds}]: appended segmentation head with in_channels={self.conv_blocks_localization[ds][-1].output_channels}")


        # New section: upscale logits and module registration
        self.upscale_logits_ops = []
        print("Debug: Initialized upscale_logits_ops list")
        
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        print(f"Debug: computed cum_upsample = {cum_upsample}")
        
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                op = Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]), mode=upsample_mode)
                self.upscale_logits_ops.append(op)
                print(f"Debug [upscale loop usl={usl}]: appended Upsample with scale_factor={op.scale_factor}")
            else:
                self.upscale_logits_ops.append(lambda x: x)
                print(f"Debug [upscale loop usl={usl}]: appended identity lambda")

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p
            print(f"Debug: restored dropout_op_kwargs['p'] to {old_dropout_p}")

        # register all modules properly
        print('##############This is conv_blocks_content #############################')
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        print("Debug: converted conv_blocks_context to ModuleList")
        print("Contents of self.conv_blocks_context:")
        for idx, conv_blocks_context in enumerate(self.conv_blocks_context):
            print(f"conv_blocks_context {idx}: {conv_blocks_context}")

        print('##############This is conv_blocks_localization #############################')
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        print("Debug: converted conv_blocks_localization to ModuleList")
        print("Contents of self.conv_blocks_localization:")
        for idx, conv_blocks_localization in enumerate(self.conv_blocks_localization):
            print(f"conv_blocks_localization {idx}: {conv_blocks_localization}")            

        print('##############This is td #############################')            
        self.td = nn.ModuleList(self.td)
        print("Debug: converted td to ModuleList")
        print("Content of self.td:")
        for idx, td in enumerate(self.td):
            print(f"td {idx}: {td}")        
        
        print('##############This is tu #############################')
        self.tu = nn.ModuleList(self.tu)
        print("Debug: converted tu to ModuleList")
        for idx, tu in enumerate(self.tu):
            print(f"tu {idx}: {tu}")
                    
        print('##############This is seg_outputs #############################')            
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        print("Debug: converted seg_outputs to ModuleList")
        print("Content of self.seg_outputs:")
        for idx, seg_outputs in enumerate(self.seg_outputs):
            print(f"seg_outputs {idx}: {seg_outputs}")        
        
        print('##############This is upscale_logits #############################')
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(self.upscale_logits_ops)
            print("Debug: converted upscale_logits_ops to ModuleList")
            print("Content of self.upscale_logits_ops:")
            for idx, upscale_logits in enumerate(self.seg_outputs):
                print(f"upscale_logits {idx}: {upscale_logits}")
            

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            print("Debug: applied weightInitializer to all modules")
            # self.apply(print_module_training_status)

    def forward(self, x):
        # print(f"Debug [forward]: input x shape = {x.shape}")
        skips = []
        # print("Debug [forward]: initialized skips list")
        seg_outputs = []
        # print("Debug [forward]: initialized seg_outputs list")
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            # print(f"Debug [forward loop d={d}]: after context conv, x shape = {x.shape}")
            skips.append(x)
            # print(f"Debug [forward loop d={d}]: appended to skips, skips length = {len(skips)}")
            if not self.convolutional_pooling:
                x = self.td[d](x)
                # print(f"Debug [forward loop d={d}]: after pooling, x shape = {x.shape}")

        x = self.conv_blocks_context[-1](x)
        # print(f"Debug [forward]: after bottleneck conv, x shape = {x.shape}")
        # raise SystemExit

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            # print(f"Debug [forward loc loop u={u}]: after upsample, x shape = {x.shape}")
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            # print(f"Debug [forward loc loop u={u}]: after concat, x shape = {x.shape}")
            x = self.conv_blocks_localization[u](x)
            # print(f"Debug [forward loc loop u={u}]: after localization conv, x shape = {x.shape}")
            so = self.final_nonlin(self.seg_outputs[u](x))
            # print(f"Debug [forward loc loop u={u}]: seg_output shape = {so.shape}")
            seg_outputs.append(so)

        if self._deep_supervision and self.do_ds:
            outs = tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
            # print(f"Debug [forward]: returning deep supervision outputs, count = {len(outs)}")
            return outs
        else:
            # print(f"Debug [forward]: returning final seg_output shape = {seg_outputs[-1].shape}")
            return seg_outputs[-1]        
        

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp
