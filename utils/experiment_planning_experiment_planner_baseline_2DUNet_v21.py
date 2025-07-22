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

from experiment_planning_common_utils import get_pool_and_conv_props
from experiment_planning_experiment_planner_baseline_2DUNet import ExperimentPlanner2D
from network_architectures_generic_UNet import Generic_UNet
from paths import *
import numpy as np



class ExperimentPlanner2D_v21(ExperimentPlanner2D):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner2D_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1_2D"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlansv2.1_plans_2D.pkl")
        self.unet_base_num_features = 32
            
    def get_properties_for_stage(self, current_spacing, original_spacing, original_shape, num_cases,
                                 num_modalities, num_classes):
        print("Entering get_properties_for_stage with:")
        print(f"  current_spacing: {current_spacing}")
        print(f"  original_spacing: {original_spacing}")
        print(f"  original_shape: {original_shape}")
        print(f"  num_cases: {num_cases}, num_modalities: {num_modalities}, num_classes: {num_classes}")

        new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)
        print(f"Calculated new_median_shape: {new_median_shape}")

        dataset_num_voxels = np.prod(new_median_shape, dtype=np.int64) * num_cases
        print(f"Calculated dataset_num_voxels: {dataset_num_voxels}")

        input_patch_size = new_median_shape[1:]
        print(f"Initial input_patch_size (new_median_shape[1:]): {input_patch_size}")

        (network_num_pool_per_axis,
         pool_op_kernel_sizes,
         conv_kernel_sizes,
         new_shp,
         shape_must_be_divisible_by) = get_pool_and_conv_props(
             current_spacing[1:], input_patch_size,
             self.unet_featuremap_min_edge_length,
             self.unet_max_numpool
         )        
        print(f"network_num_pool_per_axis: {network_num_pool_per_axis}")
        print(f"pool_op_kernel_sizes: {pool_op_kernel_sizes}")
        print(f"conv_kernel_sizes: {conv_kernel_sizes}")
        print(f"new_shp after pool/conv props: {new_shp}")
        print(f"shape_must_be_divisible_by: {shape_must_be_divisible_by}")

        # we pretend to use 30 feature maps. This will yield the same configuration as in V1. The larger memory
        # footpring of 32 vs 30 is mor ethan offset by the fp16 training. We make fp16 training default
        # Reason for 32 vs 30 feature maps is that 32 is faster in fp16 training (because multiple of 8)
        ref = Generic_UNet.use_this_for_batch_size_computation_2D * \
              Generic_UNet.DEFAULT_BATCH_SIZE_2D / 2
        print(f"Reference VRAM limit ref: {ref}")

        here = Generic_UNet.compute_approx_vram_consumption(
            new_shp,
            network_num_pool_per_axis,
            30,
            self.unet_max_num_filters,
            num_modalities,
            num_classes,
            pool_op_kernel_sizes,
            conv_per_stage=self.conv_per_stage
        )
        print(f"Initial VRAM consumption estimate here: {here}")

        while here > ref:
            print(f"here ({here}) > ref ({ref}), entering reduction loop")

            axis_to_be_reduced = np.argsort(new_shp / new_median_shape[1:])[-1]
            print(f"axis_to_be_reduced: {axis_to_be_reduced}")

            tmp = deepcopy(new_shp)
            print(f"tmp copy of new_shp before reduction: {tmp}")
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            print(f"tmp after subtracting shape_must_be_divisible_by on axis {axis_to_be_reduced}: {tmp}")

            (_, _, _, _, shape_must_be_divisible_by_new) = \
                get_pool_and_conv_props(
                    current_spacing[1:], tmp,
                    self.unet_featuremap_min_edge_length,
                    self.unet_max_numpool
                )
            print(f"shape_must_be_divisible_by_new: {shape_must_be_divisible_by_new}")

            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]
            print(f"new_shp after reduction on axis {axis_to_be_reduced}: {new_shp}")

            (network_num_pool_per_axis,
             pool_op_kernel_sizes,
             conv_kernel_sizes,
             new_shp,
             shape_must_be_divisible_by) = get_pool_and_conv_props(
                 current_spacing[1:], new_shp,
                 self.unet_featuremap_min_edge_length,
                 self.unet_max_numpool
             )
            print(f"Recomputed network_num_pool_per_axis: {network_num_pool_per_axis}")
            print(f"Recomputed pool_op_kernel_sizes: {pool_op_kernel_sizes}")
            print(f"Recomputed conv_kernel_sizes: {conv_kernel_sizes}")
            print(f"Recomputed new_shp: {new_shp}")
            print(f"Recomputed shape_must_be_divisible_by: {shape_must_be_divisible_by}")

            here = Generic_UNet.compute_approx_vram_consumption(
                new_shp,
                network_num_pool_per_axis,
                self.unet_base_num_features,
                self.unet_max_num_filters,
                num_modalities,
                num_classes,
                pool_op_kernel_sizes,
                conv_per_stage=self.conv_per_stage
            )
            print(f"Updated VRAM consumption estimate here: {here}")

        batch_size = int(np.floor(ref / here) * 2)
        print(f"Calculated batch_size: {batch_size}")

        input_patch_size = new_shp
        print(f"Final input_patch_size set to new_shp: {input_patch_size}")

        if batch_size < self.unet_min_batch_size:
            print(f"Error: batch_size {batch_size} < unet_min_batch_size {self.unet_min_batch_size}")
            raise RuntimeError("This should not happen")

        max_batch_size = np.round(
            self.batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
            np.prod(input_patch_size, dtype=np.int64)
        ).astype(int)
        print(f"Calculated max_batch_size: {max_batch_size}")

        batch_size = max(1, min(batch_size, max_batch_size))
        print(f"Clamped batch_size: {batch_size}")

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
            'do_dummy_2D_data_aug': False
        }
        print(f"Generated plan: {plan}")
        return plan

