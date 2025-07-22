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

import numpy as np
from experiment_planning_common_utils import get_pool_and_conv_props
from experiment_planning_experiment_planner_baseline_3DUNet import ExperimentPlanner
from network_architectures_generic_UNet import Generic_UNet
from paths import *


class ExperimentPlanner3D_v21(ExperimentPlanner):
    """
    Combines ExperimentPlannerPoolBasedOnSpacing and ExperimentPlannerTargetSpacingForAnisoAxis

    We also increase the base_num_features to 32. This is solely because mixed precision training with 3D convs and
    amp is A LOT faster if the number of filters is divisible by 8
    """    
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlansv2.1_plans_3D.pkl")
        self.unet_base_num_features = 32
            
    def get_target_spacing(self):
        """
        per default we use the 50th percentile=median for the target spacing. Higher spacing results in smaller data
        and thus faster and easier training. Smaller spacing results in larger data and thus longer and harder training

        For some datasets the median is not a good choice. Those are the datasets where the spacing is very anisotropic
        (for example ACDC with (10, 1.5, 1.5)). These datasets still have examples with a spacing of 5 or 6 mm in the low
        resolution axis. Choosing the median here will result in bad interpolation artifacts that can substantially
        impact performance (due to the low number of slices).
        """        
        print("Entering get_target_spacing of ExperimentPlanner3D_v21() class")
        spacings = self.dataset_properties['all_spacings']
        # print("spacings:", spacings)
        sizes = self.dataset_properties['all_sizes']
        # print("sizes:", sizes)

        target = np.percentile(np.vstack(spacings), self.target_spacing_percentile, 0)
        print("Initial target percentile spacing:", target)

        # This should be used to determine the new median shape. The old implementation is not 100% correct.
        # Fixed in 2.4
        # sizes = [np.array(i) / target * np.array(j) for i, j in zip(spacings, sizes)]        

        target_size = np.percentile(np.vstack(sizes), self.target_spacing_percentile, 0)
        print("Target size (percentile of sizes):", target_size)

        target_size_mm = np.array(target) * np.array(target_size)
        print("Target size in mm:", target_size_mm)


        # we need to identify datasets for which a different target spacing could be beneficial. These datasets have the following properties:
        # - one axis which much lower resolution than the others
        # - the lowres axis has much less voxels than the others
        # - (the size in mm of the lowres axis is also reduced)
        
        worst_spacing_axis = np.argmax(target)
        print("worst_spacing_axis:", worst_spacing_axis)

        other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
        print("other_axes:", other_axes)

        other_spacings = [target[i] for i in other_axes]
        print("other_spacings:", other_spacings)

        other_sizes = [target_size[i] for i in other_axes]
        print("other_sizes:", other_sizes)

        has_aniso_spacing = target[worst_spacing_axis] > (self.anisotropy_threshold * max(other_spacings))
        print("has_aniso_spacing:", has_aniso_spacing)

        has_aniso_voxels = target_size[worst_spacing_axis] * self.anisotropy_threshold < min(other_sizes)
        print("has_aniso_voxels:", has_aniso_voxels)
        # we don't use the last one for now
        #median_size_in_mm = target[target_size_mm] * RESAMPLING_SEPARATE_Z_ANISOTROPY_THRESHOLD < max(target_size_mm)
        
        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
            print("spacings_of_that_axis:", spacings_of_that_axis)

            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
            print("Initial target_spacing_of_that_axis:", target_spacing_of_that_axis)

            if target_spacing_of_that_axis < max(other_spacings):
                print("target_spacing_of_that_axis < max(other_spacings), adjusting")
                target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
                print("Adjusted target_spacing_of_that_axis:", target_spacing_of_that_axis)

            target[worst_spacing_axis] = target_spacing_of_that_axis
            print("Updated target[worst_spacing_axis]:", target[worst_spacing_axis])

        print("Returning target spacing:", target)
        return target
    
    
    def get_properties_for_stage(self, current_spacing, original_spacing, original_shape, num_cases,
                                    num_modalities, num_classes):
        print("Entering ExperimentPlanner3D_v21.get_properties_for_stage")
        print(f"current_spacing: {current_spacing}")
        print(f"original_spacing: {original_spacing}")
        print(f"original_shape: {original_shape}")
        print(f"num_cases: {num_cases}, num_modalities: {num_modalities}, num_classes: {num_classes}")

        """
        ExperimentPlanner configures pooling so that we pool late. Meaning that if the number of pooling per axis is
        (2, 3, 3), then the first pooling operation will always pool axes 1 and 2 and not 0, irrespective of spacing.
        This can cause a larger memory footprint, so it can be beneficial to revise this.

        Here we are pooling based on the spacing of the data.

        """
        new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)
        print(f"new_median_shape: {new_median_shape}")

        dataset_num_voxels = np.prod(new_median_shape) * num_cases
        print(f"dataset_num_voxels: {dataset_num_voxels}")

        # the next line is what we had before as a default. The patch size had the same aspect ratio as the median shape of a patient. We swapped t
        # input_patch_size = new_median_shape

        # compute how many voxels are one mm
        input_patch_size = 1 / np.array(current_spacing)
        print(f"input_patch_size (voxels/mm): {input_patch_size}")

        input_patch_size /= input_patch_size.mean()
        print(f"normalized input_patch_size: {input_patch_size}")

        # create an isotropic patch of size 512x512x512mm
        input_patch_size *= 1 / min(input_patch_size) * 512
        print(f"scaled input_patch_size to mm basis: {input_patch_size}")

        input_patch_size = np.round(input_patch_size).astype(int)
        print(f"rounded input_patch_size: {input_patch_size}")

        # clip it to the median shape of the dataset because patches larger then that make not much sense
        input_patch_size = [min(i, j) for i, j in zip(input_patch_size, new_median_shape)]
        print(f"clipped input_patch_size to median shape: {input_patch_size}")

        (network_num_pool_per_axis,
        pool_op_kernel_sizes,
        conv_kernel_sizes,
        new_shp,
        shape_must_be_divisible_by) = get_pool_and_conv_props(
            current_spacing, input_patch_size,
            self.unet_featuremap_min_edge_length,
            self.unet_max_numpool
        )
        print(f"network_num_pool_per_axis: {network_num_pool_per_axis}")
        print(f"pool_op_kernel_sizes: {pool_op_kernel_sizes}")
        print(f"conv_kernel_sizes: {conv_kernel_sizes}")
        print(f"new_shp after pool/conv props: {new_shp}")
        print(f"shape_must_be_divisible_by: {shape_must_be_divisible_by}")

        # we compute as if we were using only 30 feature maps. We can do that because fp16 training is the standard
        # now. That frees up some space. The decision to go with 32 is solely due to the speedup we get (non-multiples
        # of 8 are not supported in nvidia amp)
        ref = Generic_UNet.use_this_for_batch_size_computation_3D * self.unet_base_num_features / Generic_UNet.BASE_NUM_FEATURES_3D
        print(f"reference VRAM ref: {ref}")

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
        print(f"initial VRAM consumption here: {here}")

        while here > ref:
            print(f"here ({here}) > ref ({ref}), reducing new_shp")
            axis_to_be_reduced = np.argsort(new_shp / new_median_shape)[-1]
            print(f"axis_to_be_reduced: {axis_to_be_reduced}")

            tmp = deepcopy(new_shp)
            print(f"tmp copy of new_shp: {tmp}")
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            print(f"tmp after subtraction: {tmp}")

            (_, _, _, _, shape_must_be_divisible_by_new) = get_pool_and_conv_props(
                current_spacing, tmp,
                self.unet_featuremap_min_edge_length,
                self.unet_max_numpool
            )
            print(f"shape_must_be_divisible_by_new: {shape_must_be_divisible_by_new}")

            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]
            print(f"new_shp after reduction: {new_shp}")

            (network_num_pool_per_axis,
            pool_op_kernel_sizes,
            conv_kernel_sizes,
            new_shp,
            shape_must_be_divisible_by) = get_pool_and_conv_props(
                current_spacing, new_shp,
                self.unet_featuremap_min_edge_length,
                self.unet_max_numpool
            )
            print(f"recomputed network_num_pool_per_axis: {network_num_pool_per_axis}")
            print(f"recomputed pool_op_kernel_sizes: {pool_op_kernel_sizes}")
            print(f"recomputed conv_kernel_sizes: {conv_kernel_sizes}")
            print(f"recomputed new_shp: {new_shp}")
            print(f"recomputed shape_must_be_divisible_by: {shape_must_be_divisible_by}")

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
            print(f"updated VRAM consumption here: {here}")

        input_patch_size = new_shp
        print(f"final input_patch_size: {input_patch_size}")

        batch_size = Generic_UNet.DEFAULT_BATCH_SIZE_3D
        print(f"default batch_size: {batch_size}")
        batch_size = int(np.floor(max(ref / here, 1) * batch_size))
        print(f"scaled batch_size by VRAM ratio: {batch_size}")

        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset * dataset_num_voxels /np.prod(input_patch_size, dtype=np.int64)).astype(int)
        print(f"computed max_batch_size: {max_batch_size}")

        max_batch_size = max(max_batch_size, self.unet_min_batch_size)
        print(f"clamped max_batch_size to min_batch_size: {max_batch_size}")

        batch_size = max(1, min(batch_size, max_batch_size))
        print(f"clamped batch_size final: {batch_size}")

        do_dummy_2D_data_aug = (max(input_patch_size) / input_patch_size[0]) > self.anisotropy_threshold
        print(f"do_dummy_2D_data_aug: {do_dummy_2D_data_aug}")

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'do_dummy_2D_data_aug': do_dummy_2D_data_aug,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
        }
        print(f"returning plan: {plan}")
        return plan
