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

from torch import nn
from torch import distributed
from torch import autograd
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import subprocess

from typing import Tuple, List, Union
from skimage import io
import SimpleITK as sitk
import numpy as np
import tifffile
import nibabel as nib
from nibabel import io_orientation
from multiprocessing import Pool

from batchgenerators.utilities_file_and_folder_operations import *
from paths import *


### Utilities_task_name_id_conversion 
def convert_id_to_task_name(task_id: int):
    # print(f"Entering convert_id_to_task_name with task_id: {task_id}")
    startswith = "Task%03.0d" % task_id
    # print(f"Computed startswith prefix: {startswith}")

    if preprocessing_output_dir is not None:
        # print(f"Searching preprocessed folders in: {preprocessing_output_dir}")
        candidates_preprocessed = subdirs(preprocessing_output_dir, prefix=startswith, join=False)
    else:
        # print("No preprocessing_output_dir defined, skipping preprocessed search")
        candidates_preprocessed = []
        
    # print(f"Found candidates_preprocessed: {candidates_preprocessed}")

    if nnUNet_raw_data is not None:
        # print(f"Searching raw data folders in: {nnUNet_raw_data}")
        candidates_raw = subdirs(nnUNet_raw_data, prefix=startswith, join=False)
    else:
        # print("No nnUNet_raw_data defined, skipping raw data search")
        candidates_raw = []
        
    # print(f"Found candidates_raw: {candidates_raw}")

    if nnUNet_cropped_data is not None:
        # print(f"Searching cropped data folders in: {nnUNet_cropped_data}")
        candidates_cropped = subdirs(nnUNet_cropped_data, prefix=startswith, join=False)
    else:
        # print("No nnUNet_cropped_data defined, skipping cropped data search")
        candidates_cropped = []
        
    # print(f"Found candidates_cropped: {candidates_cropped}")

    candidates_trained_models = []
    if network_training_output_dir is not None:
        # print(f"Searching trained models in: {network_training_output_dir}")
        for m in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres']:
            model_dir = join(network_training_output_dir, m)
            if isdir(model_dir):
                # print(f"Searching in model directory: {model_dir}")
                found = subdirs(model_dir, prefix=startswith, join=False)
                # print(f"Found in {m}: {found}")
                candidates_trained_models += found
    else:
        print("No network_training_output_dir defined, skipping trained models search")
    
    # print(f"Aggregated candidates_trained_models: {candidates_trained_models}")

    all_candidates = candidates_cropped + candidates_preprocessed + candidates_raw + candidates_trained_models
    # print(f"All candidates combined: {all_candidates}")
    unique_candidates = np.unique(all_candidates)
    # print(f"Unique candidates after np.unique: {unique_candidates}")

    if len(unique_candidates) > 1:
        # print(f"Error: Multiple task names found: {unique_candidates}")
        raise RuntimeError(
            f"More than one task name found for task id {task_id}. Please correct that. (I looked in the "
            f"following folders:\n{nnUNet_raw_data}\n{preprocessing_output_dir}\n{nnUNet_cropped_data})"
        )

    if len(unique_candidates) == 0:
        # print("Error: No task name found, raising RuntimeError")
        raise RuntimeError(
            f"Could not find a task with the ID {task_id}. Make sure the requested task ID exists..."  # truncated message
        )

    result = unique_candidates[0]
    # print(f"Returning task name: {result}")
    return result


def convert_task_name_to_id(task_name: str):
    print(f"Entering convert_task_name_to_id with task_name: {task_name}")
    assert task_name.startswith("Task")
    task_id = int(task_name[4:7])
    print(f"Extracted task_id: {task_id}")
    return task_id

### Utilities_nd_softmax.py
softmax_helper = lambda x: F.softmax(x, 1)
 
 
### Utilities_distributed.py
def print_if_rank0(*args):
    print(f"Entering print_if_rank0 with args: {args}")
    rank = distributed.get_rank()
    print(f"Current distributed rank: {rank}")
    if rank == 0:
        print(*args)
        print("Message printed by rank 0")
    else:
        print("Non-zero rank, skipping print")        


class awesome_allgather_function(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        print(f"awesome_allgather_function.forward input shape: {input.shape}")
        world_size = distributed.get_world_size()
        # print(f"World size: {world_size}")
        # create a destination list for the allgather.  I'm assuming you're gathering from 3 workers.
        allgather_list = [torch.empty_like(input) for _ in range(world_size)]
        # print(f"Created empty tensors for allgather: {[t.shape for t in allgather_list]}")
        #if distributed.get_rank() == 0:
        #    import IPython;IPython.embed()
        distributed.all_gather(allgather_list, input)
        # print(f"After all_gather, gathering list shapes: {[t.shape for t in allgather_list]}")
        output = torch.cat(allgather_list, dim=0)
        print(f"Concatenated output shape: {output.shape}")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        print(f"awesome_allgather_function.backward grad_output shape: {grad_output.shape}")
        world_size = distributed.get_world_size()
        # print(f"World size in backward: {world_size}")
        grads_per_rank = grad_output.shape[0] // world_size
        # print(f"Grads per rank: {grads_per_rank}")
        rank = distributed.get_rank()
        print(f"Current rank in backward: {rank}")
        # We'll receive gradients for the entire catted forward output, so to mimic DataParallel,
        # return only the slice that corresponds to this process's input:        
        sl = slice(rank * grads_per_rank, (rank + 1) * grads_per_rank)
        # print(f"Slice for this rank: {sl}")
        grad_input = grad_output[sl]
        print(f"Returning grad_input shape: {grad_input.shape}")
        return grad_input
    
    
###  Utilities_file_ending.py   
def remove_trailing_slash(filename: str):
    print(f"Entering remove_trailing_slash with filename: {filename}")
    while filename.endswith('/'):
        # print(f"Filename endswith '/': {filename}")
        filename = filename[:-1]
        # print(f"Removed trailing slash, new filename: {filename}")
    print(f"Exiting remove_trailing_slash with result: {filename}")
    return filename


def maybe_add_0000_to_all_niigz(folder):
    print(f"Entering maybe_add_0000_to_all_niigz with folder: {folder}")
    nii_gz = subfiles(folder, suffix='.nii.gz')
    print(f"Found .nii.gz files: {nii_gz}")
    for n in nii_gz:
        n = remove_trailing_slash(n)
        if not n.endswith('_0000.nii.gz'):
            os.rename(n, n[:-7] + '_0000.nii.gz')
    print("Exiting maybe_add_0000_to_all_niigz")        


### Utilities_folder_names.py
def get_output_folder_name(model: str, task: str = None, trainer: str = None, plans: str = None, fold: int = None,
                           overwrite_training_output_dir: str = None):
    """
    Retrieves the correct output directory for the nnU-Net model described by the input parameters

    :param model:
    :param task:
    :param trainer:
    :param plans:
    :param fold:
    :param overwrite_training_output_dir:
    :return:
    """    
    print(f"Entering get_output_folder_name with model={model}, task={task}, trainer={trainer}, plans={plans}, fold={fold}, overwrite_training_output_dir={overwrite_training_output_dir}")
    assert model in ["2d", "3d_cascade_fullres", '3d_fullres', '3d_lowres'], \
        print(f"Assertion failed: invalid model type {model}") or AssertionError
    print("Model assertion passed")
    if overwrite_training_output_dir is not None:
        tr_dir = overwrite_training_output_dir
        print(f"Using overwrite_training_output_dir: {tr_dir}")
    else:
        tr_dir = network_training_output_dir
        print(f"Using network_training_output_dir: {tr_dir}")

    current = join(tr_dir, model)
    print(f"Base output folder: {current}")
    if task is not None:
        current = join(current, task)
        print(f"Added task to folder path: {current}")
        if trainer is not None and plans is not None:
            current = join(current, trainer + "__" + plans)
            print(f"Added trainer and plans: {current}")
            if fold is not None:
                fold_dir = f"fold_{fold}"
                current = join(current, fold_dir)
                print(f"Added fold to folder path: {current}")
    print(f"Exiting get_output_folder_name with result: {current}")
    return current


### Utilities_image_reorientation.py
def print_shapes(folder: str) -> None:
    print(f"Entering print_shapes with folder: {folder}")
    nii_files = subfiles(folder, suffix='.nii.gz')
    print(f"Found .nii.gz files: {nii_files}")
    for i in nii_files:
        # print(f"Reading NIfTI file: {i}")
        tmp = sitk.ReadImage(i)
        arr = sitk.GetArrayFromImage(tmp)
        print(f"Image shape: {arr.shape}, spacing: {tmp.GetSpacing()}")
    print("Exiting print_shapes")


def reorient_to_ras(image: str) -> None:
    """
    Will overwrite image!!!
    :param image:
    :return:
    """
    print(f"Entering reorient_to_ras with image: {image}")    
    assert image.endswith('.nii.gz')
    origaffine_pkl = image[:-7] + '_originalAffine.pkl'
    print(f"Original affine pickle path: {origaffine_pkl}")    
    if not isfile(origaffine_pkl):
        print("No existing original affine pickle found, proceeding")        
        img = nib.load(image)
        print(f"Loaded image, affine shape: {img.affine.shape}")        
        original_affine = img.affine
        original_axcode = nib.aff2axcodes(img.affine)
        img = img.as_reoriented(io_orientation(img.affine))
        print("Reoriented image to RAS orientation")        
        new_axcode = nib.aff2axcodes(img.affine)
        print(image.split('/')[-1], 'original axcode', original_axcode, 'now (should be ras)', new_axcode)
        nib.save(img, image)
        save_pickle((original_affine, original_axcode), origaffine_pkl)


def revert_reorientation(image: str) -> None:
    print(f"Entering revert_reorientation with image: {image}")
    assert image.endswith('.nii.gz')
    expected_pkl = image[:-7] + '_originalAffine.pkl'
    print(f"Expected pickle path for original affine: {expected_pkl}")
    assert isfile(expected_pkl), 'Must have a file with the original affine, as created by ' \
                                 'reorient_to_ras. Expected filename: %s' % \
                                 expected_pkl
    original_affine, original_axcode = load_pickle(image[:-7] + '_originalAffine.pkl')
    img = nib.load(image)
    before_revert = nib.aff2axcodes(img.affine)
    img = img.as_reoriented(io_orientation(original_affine))
    after_revert = nib.aff2axcodes(img.affine)
    print('before revert', before_revert, 'after revert', after_revert)
    restored_affine = img.affine
    assert np.all(np.isclose(original_affine, restored_affine)), 'restored affine does not match original affine, ' \
                                                                 'aborting!'
    nib.save(img, image)
    os.remove(expected_pkl)


def reorient_all_images_in_folder_to_ras(folder: str, num_processes: int = 8):
    p = Pool(num_processes)
    nii_files = subfiles(folder, suffix='.nii.gz', join=True)
    p.map(reorient_to_ras, nii_files)
    p.close()
    p.join()


def revert_orientation_on_all_images_in_folder(folder: str, num_processes: int = 8):
    p = Pool(num_processes)
    nii_files = subfiles(folder, suffix='.nii.gz', join=True)
    p.map(revert_reorientation, nii_files)
    p.close()
    p.join()
    
    
### Utilities_file_conversions.py
def convert_2d_image_to_nifti(input_filename: str, output_filename_truncated: str, spacing=(999, 1, 1),
                              transform=None, is_seg: bool = False) -> None:
    """
    Reads an image (must be a format that it recognized by skimage.io.imread) and converts it into a series of niftis.
    The image can have an arbitrary number of input channels which will be exported separately (_0000.nii.gz,
    _0001.nii.gz, etc for images and only .nii.gz for seg).
    Spacing can be ignored most of the time.
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!

    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net

    If Transform is not None it will be applied to the image after loading.

    Segmentations will be converted to np.uint32!

    :param is_seg:
    :param transform:
    :param input_filename:
    :param output_filename_truncated: do not use a file ending for this one! Example: output_name='./converted/image1'. This
    function will add the suffix (_0000) and file ending (.nii.gz) for you.
    :param spacing:
    :return:
    """
    img = io.imread(input_filename)

    if transform is not None:
        img = transform(img)

    if len(img.shape) == 2:  # 2d image with no color channels
        img = img[None, None]  # add dimensions
    else:
        assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
        # we assume that the color channel is the last dimension. Transpose it to be in first
        img = img.transpose((2, 0, 1))
        # add third dimension
        img = img[:, None]

    # image is now (c, x, x, z) where x=1 since it's 2d
    if is_seg:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'

    for j, i in enumerate(img):

        if is_seg:
            i = i.astype(np.uint32)

        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(list(spacing)[::-1])
        if not is_seg:
            sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")


def convert_3d_tiff_to_nifti(filenames: List[str], output_name: str, spacing: Union[tuple, list], transform=None, is_seg=False) -> None:
    """
    filenames must be a list of strings, each pointing to a separate 3d tiff file. One file per modality. If your data
    only has one imaging modality, simply pass a list with only a single entry

    Files in filenames must be readable with

    Note: we always only pass one file into tifffile.imread, not multiple (even though it supports it). This is because
    I am not familiar enough with this functionality and would like to have control over what happens.

    If Transform is not None it will be applied to the image after loading.

    :param transform:
    :param filenames:
    :param output_name:
    :param spacing:
    :return:
    """
    if is_seg:
        assert len(filenames) == 1

    for j, i in enumerate(filenames):
        img = tifffile.imread(i)

        if transform is not None:
            img = transform(img)

        itk_img = sitk.GetImageFromArray(img)
        itk_img.SetSpacing(list(spacing)[::-1])

        if not is_seg:
            sitk.WriteImage(itk_img, output_name + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_name + ".nii.gz")


def convert_2d_segmentation_nifti_to_img(nifti_file: str, output_filename: str, transform=None, export_dtype=np.uint8):
    img = sitk.GetArrayFromImage(sitk.ReadImage(nifti_file))
    assert img.shape[0] == 1, "This function can only export 2D segmentations!"
    img = img[0]
    if transform is not None:
        img = transform(img)

    io.imsave(output_filename, img.astype(export_dtype), check_contrast=False)


def convert_3d_segmentation_nifti_to_tiff(nifti_file: str, output_filename: str, transform=None, export_dtype=np.uint8):
    img = sitk.GetArrayFromImage(sitk.ReadImage(nifti_file))
    assert len(img.shape) == 3, "This function can only export 3D segmentations!"
    if transform is not None:
        img = transform(img)

    tifffile.imsave(output_filename, img.astype(export_dtype))
    
    
###   Utilities_one_hot_encoding.py
def to_one_hot(seg, all_seg_labels=None):
    if all_seg_labels is None:
        all_seg_labels = np.unique(seg)
    result = np.zeros((len(all_seg_labels), *seg.shape), dtype=seg.dtype)
    for i, l in enumerate(all_seg_labels):
        result[i][seg == l] = 1
    return result  


### Utilities_overlay_plots.py
color_cycle = (
    "000000",
    "4363d8",
    "f58231",
    "3cb44b",
    "e6194B",
    "911eb4",
    "ffe119",
    "bfef45",
    "42d4f4",
    "f032e6",
    "000075",
    "9A6324",
    "808000",
    "800000",
    "469990",
)


def hex_to_rgb(hex: str):
    assert len(hex) == 6
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


def generate_overlay(input_image: np.ndarray, segmentation: np.ndarray, mapping: dict = None, color_cycle=color_cycle,
                     overlay_intensity=0.6):
    """
    image must be a color image, so last dimension must be 3. if image is grayscale, tile it first!
    Segmentation must be label map of same shape as image (w/o color channels)
    mapping can be label_id -> idx_in_cycle or None

    returned image is scaled to [0, 255]!!!
    """
    # assert len(image.shape) == len(segmentation.shape)
    # assert all([i == j for i, j in zip(image.shape, segmentation.shape)])

    # create a copy of image
    image = np.copy(input_image)

    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], (1, 1, 3))
    elif len(image.shape) == 3:
        assert image.shape[2] == 3, 'if 3d image is given the last dimension must be the color channels ' \
                                    '(3 channels). Only 2D images are supported'

    else:
        raise RuntimeError("unexpected image shape. only 2D images and 2D images with color channels (color in "
                           "last dimension) are supported")

    # rescale image to [0, 255]
    image = image - image.min()
    image = image / image.max() * 255

    # create output

    if mapping is None:
        uniques = np.unique(segmentation)
        mapping = {i: c for c, i in enumerate(uniques)}

    for l in mapping.keys():
        image[segmentation == l] += overlay_intensity * np.array(hex_to_rgb(color_cycle[mapping[l]]))

    # rescale result to [0, 255]
    image = image / image.max() * 255
    return image.astype(np.uint8)


def plot_overlay(image_file: str, segmentation_file: str, output_file: str, overlay_intensity: float = 0.6):
    import matplotlib.pyplot as plt

    image = sitk.GetArrayFromImage(sitk.ReadImage(image_file))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(segmentation_file))
    assert all([i == j for i, j in zip(image.shape, seg.shape)]), "image and seg do not have the same shape: %s, %s" % (
        image_file, segmentation_file)

    assert len(image.shape) == 3, 'only 3D images/segs are supported'

    fg_mask = seg != 0
    fg_per_slice = fg_mask.sum((1, 2))
    selected_slice = np.argmax(fg_per_slice)

    overlay = generate_overlay(image[selected_slice], seg[selected_slice], overlay_intensity=overlay_intensity)

    plt.imsave(output_file, overlay)


def plot_overlay_preprocessed(case_file: str, output_file: str, overlay_intensity: float = 0.6, modality_index=0):
    import matplotlib.pyplot as plt
    data = np.load(case_file)['data']

    assert modality_index < (data.shape[0] - 1), 'This dataset only supports modality index up to %d' % (data.shape[0] - 2)

    image = data[modality_index]
    seg = data[-1]
    seg[seg < 0] = 0

    fg_mask = seg > 0
    fg_per_slice = fg_mask.sum((1, 2))
    selected_slice = np.argmax(fg_per_slice)

    overlay = generate_overlay(image[selected_slice], seg[selected_slice], overlay_intensity=overlay_intensity)

    plt.imsave(output_file, overlay)


def multiprocessing_plot_overlay(list_of_image_files, list_of_seg_files, list_of_output_files, overlay_intensity,
                                 num_processes=8):
    p = Pool(num_processes)
    r = p.starmap_async(plot_overlay, zip(
        list_of_image_files, list_of_seg_files, list_of_output_files, [overlay_intensity] * len(list_of_output_files)
    ))
    r.get()
    p.close()
    p.join()


def multiprocessing_plot_overlay_preprocessed(list_of_case_files, list_of_output_files, overlay_intensity,
                                 num_processes=8, modality_index=0):
    p = Pool(num_processes)
    r = p.starmap_async(plot_overlay_preprocessed, zip(
        list_of_case_files, list_of_output_files, [overlay_intensity] * len(list_of_output_files),
        [modality_index] * len(list_of_output_files)
    ))
    r.get()
    p.close()
    p.join()


def generate_overlays_for_task(task_name_or_id, output_folder, num_processes=8, modality_idx=0, use_preprocessed=True,
                               data_identifier=default_data_identifier):
    if isinstance(task_name_or_id, str):
        if not task_name_or_id.startswith("Task"):
            task_name_or_id = int(task_name_or_id)
            task_name = convert_id_to_task_name(task_name_or_id)
        else:
            task_name = task_name_or_id
    else:
        task_name = convert_id_to_task_name(int(task_name_or_id))

    if not use_preprocessed:
        folder = join(nnUNet_raw_data, task_name)

        identifiers = [i[:-7] for i in subfiles(join(folder, 'labelsTr'), suffix='.nii.gz', join=False)]

        image_files = [join(folder, 'imagesTr', i + "_%04.0d.nii.gz" % modality_idx) for i in identifiers]
        seg_files = [join(folder, 'labelsTr', i + ".nii.gz") for i in identifiers]

        assert all([isfile(i) for i in image_files])
        assert all([isfile(i) for i in seg_files])

        maybe_mkdir_p(output_folder)
        output_files = [join(output_folder, i + '.png') for i in identifiers]
        multiprocessing_plot_overlay(image_files, seg_files, output_files, 0.6, num_processes)
    else:
        folder = join(preprocessing_output_dir, task_name)
        if not isdir(folder): raise RuntimeError("run preprocessing for that task first")
        matching_folders = subdirs(folder, prefix=data_identifier + "_stage")
        if len(matching_folders) == 0: "run preprocessing for that task first (use default experiment planner!)"
        matching_folders.sort()
        folder = matching_folders[-1]
        identifiers = [i[:-4] for i in subfiles(folder, suffix='.npz', join=False)]
        maybe_mkdir_p(output_folder)
        output_files = [join(output_folder, i + '.png') for i in identifiers]
        image_files = [join(folder, i + ".npz") for i in identifiers]
        maybe_mkdir_p(output_folder)
        multiprocessing_plot_overlay_preprocessed(image_files, output_files, overlay_intensity=0.6,
                                                  num_processes=num_processes, modality_index=modality_idx)


def entry_point_generate_overlay():
    import argparse
    parser = argparse.ArgumentParser("Plots png overlays of the slice with the most foreground. Note that this "
                                     "disregards spacing information!")
    parser.add_argument('-t', type=str, help="task name or task ID", required=True)
    parser.add_argument('-o', type=str, help="output folder", required=True)
    parser.add_argument('-num_processes', type=int, default=8, required=False, help="number of processes used. Default: 8")
    parser.add_argument('-modality_idx', type=int, default=0, required=False,
                        help="modality index used (0 = _0000.nii.gz). Default: 0")
    parser.add_argument('--use_raw', action='store_true', required=False, help="if set then we use raw data. else "
                                                                               "we use preprocessed")
    args = parser.parse_args()

    generate_overlays_for_task(args.t, args.o, args.num_processes, args.modality_idx, use_preprocessed=not args.use_raw)
    
    
### Utilities_random_stuff.py
class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass    

### Utilities_recursive_delete_npz.py
def recursive_delete_npz(current_directory: str):
    npz_files = subfiles(current_directory, join=True, suffix=".npz")
    npz_files = [i for i in npz_files if not i.endswith("segFromPrevStage.npz")] # to be extra safe
    _ = [os.remove(i) for i in npz_files]
    for d in subdirs(current_directory, join=False):
        if d != "pred_next_stage":
            recursive_delete_npz(join(current_directory, d))
    
    
### Utilities_recursive_rename_taskXX_to_taskXXX.py    
def recursive_rename(folder):
    s = subdirs(folder, join=False)
    for ss in s:
        if ss.startswith("Task") and ss.find("_") == 6:
            task_id = int(ss[4:6])
            name = ss[7:]
            os.rename(join(folder, ss), join(folder, "Task%03.0d_" % task_id + name))
    s = subdirs(folder, join=True)
    for ss in s:
        recursive_rename(ss)
        
### Utilities_set_n_proc_DA.py
def get_allowed_n_proc_DA():
    hostname = subprocess.getoutput(['hostname'])

    if 'nnUNet_n_proc_DA' in os.environ.keys():
        return int(os.environ['nnUNet_n_proc_DA'])

    if hostname in ['hdf19-gpu16', 'hdf19-gpu17', 'e230-AMDworkstation']:
        return 16

    if hostname in ['Fabian',]:
        return 12

    if hostname.startswith('hdf19-gpu') or hostname.startswith('e071-gpu'):
        return 12
    elif hostname.startswith('e230-dgx1'):
        return 10
    elif hostname.startswith('hdf18-gpu') or hostname.startswith('e132-comp'):
        return 16
    elif hostname.startswith('e230-dgx2'):
        return 6
    elif hostname.startswith('e230-dgxa100-'):
        return 32
    else:
        return None
    
    
### Utilities_sitk_stuff.py
def copy_geometry(image: sitk.Image, ref: sitk.Image):
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image            

### Utilities_tensor_utilities.py
def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def mean_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.mean(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.mean(int(ax))
    return inp


def flip(x, dim):
    """
    flips the tensor at dimension dim (mirroring!)
    :param x:
    :param dim:
    :return:
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

### Utilities_to_torch.py
def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data