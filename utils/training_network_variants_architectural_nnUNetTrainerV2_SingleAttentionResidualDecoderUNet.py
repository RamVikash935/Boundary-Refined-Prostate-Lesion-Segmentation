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

from typing import Tuple

import numpy as np
import torch
from Every_part_of_network_architectures_EASAG_ResidualDecoder_UNet import ResidualDecoderSingleAttentionUNet, get_default_network_config
from network_architectures_initialization import InitWeights_He
from training_network_nnUNetTrainer import nnUNetTrainer
from training_network_nnUNetTrainerV2 import nnUNetTrainerV2
from utilities import softmax_helper


class nnUNetTrainerV2_SpatialSingleAttention_n_Residual_DecoderUNet(nnUNetTrainerV2):
    """Network Trainer for Fully Residual UNet"""
    def initialize_network(self):
        # print("DEBUG: Entered initialize_network method of nnUNetTrainerV2_SpatialSingleAttentionUNet_class")
        if self.threeD:
            cfg = get_default_network_config(3, None, norm_type="in")

        else:
            cfg = get_default_network_config(1, None, norm_type="in")

        # Note can hard code this so that the resulting architecture matches the general structure of the generic UNet

        stage_plans = self.plans['plans_per_stage'][self.stage]  # check if this needs to be changed
        conv_kernel_sizes = stage_plans['conv_kernel_sizes']  # check if this need to be changed
        blocks_per_stage_encoder = stage_plans['num_blocks_encoder']  # check if this needs to be changed
        blocks_per_stage_decoder = stage_plans['num_blocks_decoder']  # check if this need to be changed and what to
        pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        self.network = ResidualDecoderSingleAttentionUNet(input_channels=self.num_input_channels, base_num_features=self.base_num_features,
                                                  num_blocks_per_stage_encoder=blocks_per_stage_encoder, feat_map_mul_on_downscale=2,
                                                  pool_op_kernel_sizes=pool_op_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes,
                                                  props=cfg, num_classes=self.num_classes,
                                                  num_blocks_per_stage_decoder=blocks_per_stage_decoder, deep_supervision=True,
                                                  upscale_logits=False, max_features=320, initializer=InitWeights_He(1e-2))
        
        # print("DEBUG: Network initialized: ", self.network)
        
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def setup_DA_params(self):
        """
        net_num_pool_op_kernel_sizes is different in resunet
        """
        super().setup_DA_params()
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes[1:]), axis=0))[:-1]

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 force_separate_z: bool = None, interpolation_order: int = 3, interpolation_order_z=0,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.validate(self, do_mirroring=do_mirroring, use_sliding_window=use_sliding_window,
                                     step_size=step_size, save_softmax=save_softmax, use_gaussian=use_gaussian,
                                     overwrite=overwrite, validation_folder_name=validation_folder_name,
                                     debug=debug, all_in_gpu=all_in_gpu,
                                     segmentation_export_kwargs=segmentation_export_kwargs,
                                     run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.decoder.deep_supervision = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.predict_preprocessed_data_return_seg_and_softmax(self, data, do_mirroring=do_mirroring,
                                                                             mirror_axes=mirror_axes,
                                                                             use_sliding_window=use_sliding_window,
                                                                             step_size=step_size,
                                                                             use_gaussian=use_gaussian,
                                                                             pad_border_mode=pad_border_mode,
                                                                             pad_kwargs=pad_kwargs,
                                                                             all_in_gpu=all_in_gpu,
                                                                             verbose=verbose,
                                                                             mixed_precision=mixed_precision)
        self.network.decoder.deep_supervision = ds
        return ret

    def run_training(self):
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = True
        ret = nnUNetTrainer.run_training(self)
        self.network.decoder.deep_supervision = ds
        return ret













