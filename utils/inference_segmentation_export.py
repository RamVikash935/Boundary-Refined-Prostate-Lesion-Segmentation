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

import sys
from copy import deepcopy
from typing import Union, Tuple

import numpy as np
import SimpleITK as sitk
from batchgenerators.augmentations_utils import resize_segmentation
from preprocessing_preprocessing import get_lowres_axis, get_do_separate_z, resample_data_or_seg
from batchgenerators.utilities_file_and_folder_operations import *


def save_segmentation_nifti_from_softmax(segmentation_softmax: Union[str, np.ndarray], out_fname: str,
                                         properties_dict: dict, order: int = 1,
                                         region_class_order: Tuple[Tuple[int]] = None,
                                         seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                         resampled_npz_fname: str = None,
                                         non_postprocessed_fname: str = None, force_separate_z: bool = None,
                                         interpolation_order_z: int = 0, verbose: bool = True):
    """
    This is a utility for writing segmentations to nifty and npz. It requires the data to have been preprocessed by
    GenericPreprocessor because it depends on the property dictionary output (dct) to know the geometry of the original
    data. segmentation_softmax does not have to have the same size in pixels as the original data, it will be
    resampled to match that. This is generally useful because the spacings our networks operate on are most of the time
    not the native spacings of the image data.
    If seg_postprogess_fn is not None then seg_postprogess_fnseg_postprogess_fn(segmentation, *seg_postprocess_args)
    will be called before nifty export
    There is a problem with python process communication that prevents us from communicating objects
    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
    patching system python code.) We circumvent that problem here by saving softmax_pred to a npy file that will
    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
    filename or np.ndarray for segmentation_softmax and will handle this automatically
    :param segmentation_softmax:
    :param out_fname:
    :param properties_dict:
    :param order:
    :param region_class_order:
    :param seg_postprogess_fn:
    :param seg_postprocess_args:
    :param resampled_npz_fname:
    :param non_postprocessed_fname:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately. Do not touch unless you know what you are doing
    :param interpolation_order_z: if separate z resampling is done then this is the order for resampling in z
    :param verbose:
    :return:
    """
    if verbose: print(f"[DEBUG] force_separate_z: {force_separate_z}, interpolation order: {order}")

    # Load softmax if path
    if isinstance(segmentation_softmax, str):
        print(f"[DEBUG] segmentation_softmax is a filepath: {segmentation_softmax}")
        assert isfile(segmentation_softmax), "Softmax file not found"
        del_file = deepcopy(segmentation_softmax)
        if segmentation_softmax.endswith('.npy'):
            segmentation_softmax = np.load(segmentation_softmax)
            print("[DEBUG] Loaded .npy softmax")
        elif segmentation_softmax.endswith('.npz'):
            segmentation_softmax = np.load(segmentation_softmax)['softmax']
            print("[DEBUG] Loaded .npz softmax")
        os.remove(del_file)
        print(f"[DEBUG] Deleted temporary file: {del_file}")

    # Check if resampling needed
    current_shape = segmentation_softmax.shape
    print(f"[DEBUG] current_shape: {current_shape}")
    shape_after = properties_dict.get('size_after_cropping')
    shape_before = properties_dict.get('original_size_of_raw_data')
    print(f"[DEBUG] target shape_after_cropping: {shape_after}, original_size: {shape_before}")

    if np.any([i != j for i, j in zip(current_shape[1:], shape_after)]):
        print("[DEBUG] Resampling required")
        # decide separate z
        if force_separate_z is None:
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
                print("[DEBUG] separate_z from original_spacing")
            elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
                print("[DEBUG] separate_z from spacing_after_resampling")
            else:
                do_separate_z = False
                lowres_axis = None
                print("[DEBUG] no separate_z by default")
        else:
            do_separate_z = force_separate_z
            lowres_axis = get_lowres_axis(properties_dict.get('original_spacing')) if do_separate_z else None
            print(f"[DEBUG] forced separate_z={do_separate_z}, lowres_axis={lowres_axis}")

        if lowres_axis is not None and len(lowres_axis) != 1:
            do_separate_z = False
            print("[DEBUG] lowres_axis length !=1, disabling separate_z")

        if verbose: print(f"[DEBUG] separate_z: {do_separate_z}, lowres_axis: {lowres_axis}")
        seg_old_spacing = resample_data_or_seg(
            segmentation_softmax,
            shape_after,
            is_seg=False,
            axis=lowres_axis,
            order=order,
            do_separate_z=do_separate_z,
            order_z=interpolation_order_z
        )
        print("[DEBUG] Completed resample_data_or_seg")
    else:
        if verbose: print("[DEBUG] no resampling necessary")
        seg_old_spacing = segmentation_softmax

    # Save intermediate NPZ if requested
    if resampled_npz_fname is not None:
        np.savez_compressed(resampled_npz_fname, softmax=seg_old_spacing.astype(np.float16))
        print(f"[DEBUG] Saved compressed softmax to {resampled_npz_fname}")
        if region_class_order is not None:
            properties_dict['regions_class_order'] = region_class_order
            print(f"[DEBUG] Updated properties_dict with region_class_order: {region_class_order}")
        save_pickle(properties_dict, resampled_npz_fname[:-4] + ".pkl")
        print(f"[DEBUG] Saved properties_dict pickle to {resampled_npz_fname[:-4] + '.pkl'}")

    # Convert to label map
    if region_class_order is None:
        seg_old_spacing = seg_old_spacing.argmax(0)
        print("[DEBUG] Converted softmax to labels via argmax")
    else:
        print(f"[DEBUG] Applying custom region_class_order: {region_class_order}")
        seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
        for i, c in enumerate(region_class_order):
            seg_old_spacing_final[seg_old_spacing[i] > 0.5] = c
            print(f"[DEBUG] Region {i} thresholded >0.5 to class {c}")
        seg_old_spacing = seg_old_spacing_final
        print("[DEBUG] Completed custom region mapping")

    # Paste into original-sized volume
    bbox = properties_dict.get('crop_bbox')
    print(f"[DEBUG] crop_bbox: {bbox}")
    if bbox is not None:
        seg_old_size = np.zeros(shape_before, dtype=np.uint8)
        print(f"[DEBUG] Initialized empty volume of size {shape_before}")
        for c in range(3):
            bbox[c][1] = min(bbox[c][0] + seg_old_spacing.shape[c], shape_before[c])
            print(f"[DEBUG] Adjusted bbox axis {c}: {bbox[c]}")
        seg_old_size[
            bbox[0][0]:bbox[0][1],
            bbox[1][0]:bbox[1][1],
            bbox[2][0]:bbox[2][1]
        ] = seg_old_spacing
        print("[DEBUG] Pasted segmentation into full volume")
    else:
        seg_old_size = seg_old_spacing
        print("[DEBUG] No cropping; using seg_old_spacing as full volume")

    # Optional post-processing
    if seg_postprogess_fn is not None:
        seg_old_size_postprocessed = seg_postprogess_fn(np.copy(seg_old_size), *seg_postprocess_args)
        print("[DEBUG] Applied seg_postprogess_fn")
    else:
        seg_old_size_postprocessed = seg_old_size
        print("[DEBUG] No post-processing applied")

    # Write final NIfTI
    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.uint8))
    print("[DEBUG] Converted postprocessed array to ITK image")
    seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict['itk_direction'])
    print("[DEBUG] Set ITK image metadata (spacing, origin, direction)")
    sitk.WriteImage(seg_resized_itk, out_fname)
    print(f"[DEBUG] Wrote segmentation NIfTI to {out_fname}")

    # Optionally save non-postprocessed version
    if non_postprocessed_fname is not None and seg_postprogess_fn is not None:
        seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
        print("[DEBUG] Converted non-postprocessed array to ITK image")
        seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
        seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
        seg_resized_itk.SetDirection(properties_dict['itk_direction'])
        print("[DEBUG] Set metadata for non-postprocessed ITK image")
        sitk.WriteImage(seg_resized_itk, non_postprocessed_fname)
        print(f"[DEBUG] Wrote non-postprocessed NIfTI to {non_postprocessed_fname}")



def save_segmentation_nifti(segmentation, out_fname, dct, order=1, force_separate_z=None, order_z=0, verbose: bool = False):
    """
    faster and uses less ram than save_segmentation_nifti_from_softmax, but maybe less precise and also does not support
    softmax export (which is needed for ensembling). So it's a niche function that may be useful in some cases.
    :param segmentation:
    :param out_fname:
    :param dct:
    :param order:
    :param force_separate_z:
    :return:
    """
    print(f"[DEBUG] force_separate_z: {force_separate_z}, interpolation order: {order}")
    # suppress output    
    if not verbose:
        print("[DEBUG] Suppressing stdout")
        sys.stdout = open(os.devnull, 'w')

    # Load if filepath
    if isinstance(segmentation, str):
        print(f"[DEBUG] segmentation is filepath: {segmentation}")
        assert isfile(segmentation), "Segmentation file not found"
        del_file = deepcopy(segmentation)
        segmentation = np.load(segmentation)
        print("[DEBUG] Loaded segmentation array from file")
        os.remove(del_file)
        print(f"[DEBUG] Deleted temporary file: {del_file}")

    # Determine shapes
    # first resample, then put result into bbox of cropping, then save    
    current_shape = segmentation.shape
    shape_after = dct.get('size_after_cropping')
    shape_before = dct.get('original_size_of_raw_data')
    print(f"[DEBUG] current_shape: {current_shape}, target shape_after: {shape_after}")
    # current_spacing = dct.get('spacing_after_resampling')
    # original_spacing = dct.get('original_spacing')    

    # Resample if necessary
    if np.any(np.array(current_shape) != np.array(shape_after)):
        print("[DEBUG] Resampling needed")
        if order == 0:
            seg_old_spacing = resize_segmentation(segmentation, shape_after, 0)
            print("[DEBUG] Resampled with nearest-neighbor (order=0)")
        else:
            # decide separate z
            if force_separate_z is None:
                if get_do_separate_z(dct.get('original_spacing')):
                    do_separate_z = True
                    lowres_axis = get_lowres_axis(dct.get('original_spacing'))
                    print("[DEBUG] separate_z from original_spacing")
                elif get_do_separate_z(dct.get('spacing_after_resampling')):
                    do_separate_z = True
                    lowres_axis = get_lowres_axis(dct.get('spacing_after_resampling'))
                    print("[DEBUG] separate_z from spacing_after_resampling")
                else:
                    do_separate_z = False
                    lowres_axis = None
                    print("[DEBUG] no separate_z by default")
            else:
                do_separate_z = force_separate_z
                lowres_axis = get_lowres_axis(dct.get('original_spacing')) if do_separate_z else None
                print(f"[DEBUG] forced separate_z={do_separate_z}, lowres_axis={lowres_axis}")

            print(f"[DEBUG] Calling resample_data_or_seg with order={order}, order_z={order_z}")
            seg_old_spacing = resample_data_or_seg(
                segmentation[None], shape_after, is_seg=True,
                axis=lowres_axis, order=order,
                do_separate_z=do_separate_z, order_z=order_z
            )[0]
            print("[DEBUG] Completed resample_data_or_seg")
    else:
        print("[DEBUG] No resampling necessary")
        seg_old_spacing = segmentation

    # Apply crop bbox
    bbox = dct.get('crop_bbox')
    print(f"[DEBUG] crop_bbox: {bbox}")
    if bbox is not None:
        seg_old_size = np.zeros(shape_before)
        print(f"[DEBUG] Initialized empty full-volume array of shape {shape_before}")
        for c in range(3):
            bbox[c][1] = min(bbox[c][0] + seg_old_spacing.shape[c], shape_before[c])
            print(f"[DEBUG] Adjusted bbox axis {c}: {bbox[c]}")
        seg_old_size[
            bbox[0][0]:bbox[0][1],
            bbox[1][0]:bbox[1][1],
            bbox[2][0]:bbox[2][1]
        ] = seg_old_spacing
        print("[DEBUG] Pasted resampled segmentation into full-volume array")
    else:
        seg_old_size = seg_old_spacing
        print("[DEBUG] No cropping applied, using resampled array directly")

    # Write NIfTI
    seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
    print("[DEBUG] Converted array to ITK image")
    seg_resized_itk.SetSpacing(dct['itk_spacing'])
    seg_resized_itk.SetOrigin(dct['itk_origin'])
    seg_resized_itk.SetDirection(dct['itk_direction'])
    print("[DEBUG] Set ITK image metadata (spacing, origin, direction)")
    sitk.WriteImage(seg_resized_itk, out_fname)
    print(f"[DEBUG] Wrote segmentation NIfTI to {out_fname}")

    # Restore stdout
    if not verbose:
        sys.stdout = sys.__stdout__
        print("[DEBUG] Restored stdout")

