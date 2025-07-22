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

import SimpleITK as sitk
import numpy as np
import shutil
from batchgenerators.utilities_file_and_folder_operations import *
from multiprocessing import Pool
from collections import OrderedDict



def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    # print(f"Entering create_nonzero_mask with data.shape: {data.shape}")
    assert len(data.shape) in (3, 4), \
        print(f"Assertion failed: data must have shape (C, X, Y, Z) or (C, X, Y), got {data.shape}") or AssertionError
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    # print(f"Initialized nonzero_mask with shape: {nonzero_mask.shape}")
    for c in range(data.shape[0]):
        # print(f"Processing channel {c}")
        this_mask = data[c] != 0
        # print(f"this_mask for channel {c} has {np.sum(this_mask)} nonzero voxels")
        nonzero_mask = nonzero_mask | this_mask
        # print(f"Updated nonzero_mask after channel {c}")
    nonzero_mask = binary_fill_holes(nonzero_mask)
    # print(f"Applied binary_fill_holes, nonzero_mask shape: {nonzero_mask.shape}")
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    # print(f"Entering get_bbox_from_mask with mask.shape: {mask.shape}, outside_value: {outside_value}")
    mask_voxel_coords = np.where(mask != outside_value)
    # print(f"mask_voxel_coords lengths: {[len(coords) for coords in mask_voxel_coords]}")
    minzidx = int(np.min(mask_voxel_coords[0])); 
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1; 
    minxidx = int(np.min(mask_voxel_coords[1])); 
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1; 
    minyidx = int(np.min(mask_voxel_coords[2])); 
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1; 
    bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
    # print(f"Returning bbox: {bbox}")
    return bbox


def crop_to_bbox(image, bbox):
    # print(f"Entering crop_to_bbox with image.shape: {image.shape}, bbox: {bbox}")
    assert len(image.shape) == 3, \
        print(f"Assertion failed: only supports 3D images, got shape {image.shape}") or AssertionError
    resizer = (
        slice(bbox[0][0], bbox[0][1]),
        slice(bbox[1][0], bbox[1][1]),
        slice(bbox[2][0], bbox[2][1])
    )
    # print(f"Using resizer slices: {resizer}")
    cropped = image[resizer]
    # print(f"Cropped image shape: {cropped.shape}")
    return cropped


def get_case_identifier(case):
    # print(f"Entering get_case_identifier with case: {case}")
    case_identifier = case[0].split("/")[-1].split(".nii.gz")[0][:-5]
    # print(f"Extracted case_identifier: {case_identifier}")
    return case_identifier


def get_case_identifier_from_npz(case):
    # print(f"Entering get_case_identifier_from_npz with case: {case}")
    case_identifier = case.split("/")[-1][:-4]
    # print(f"Extracted case_identifier: {case_identifier}")
    return case_identifier


def load_case_from_list_of_files(data_files, seg_file=None):
    # print(f"Entering load_case_from_list_of_files with data_files: {data_files}, seg_file: {seg_file}")
    assert isinstance(data_files, (list, tuple)), \
        print(f"Assertion failed: data_files must be list or tuple, got {type(data_files)}") or AssertionError
    properties = OrderedDict()
    # print("Initialized properties OrderedDict")

    data_itk = []
    for f in data_files:
        # print(f"Reading image with SimpleITK: {f}")
        img = sitk.ReadImage(f)
        data_itk.append(img)
        # print(f"Read image, size: {img.GetSize()}, spacing: {img.GetSpacing()}")

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2,1,0]]
    # print(f"Saved original_size_of_raw_data: {properties['original_size_of_raw_data']}")
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2,1,0]]
    # print(f"Saved original_spacing: {properties['original_spacing']}")
    properties["list_of_data_files"] = data_files
    # print(f"Saved list_of_data_files: {properties['list_of_data_files']}")
    properties["seg_file"] = seg_file
    # print(f"Saved seg_file: {properties['seg_file']}")

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])
    # print(f"Constructed data_npy with shape: {data_npy.shape}")
    if seg_file is not None:
        # print(f"Reading segmentation image: {seg_file}")
        seg_itk = sitk.ReadImage(seg_file)
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
        # print(f"Constructed seg_npy with shape: {seg_npy.shape}")
    else:
        seg_npy = None
        # print("No seg_file provided, seg_npy set to None")

    # print(f"Returning data_npy dtype: {data_npy.dtype}, seg_npy: {type(seg_npy)}, properties keys: {list(properties.keys())}")
    return data_npy.astype(np.float32), seg_npy, properties



def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """
    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        # print(f"Initializing cropped_seg list")
        for c in range(seg.shape[0]):
            # print(f"Cropping seg channel {c}")
            cropped = crop_to_bbox(seg[c], bbox)
            cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)

    nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None]
    if seg is not None:
        # print("Applying nonzero_label to seg outside mask")
        seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    else:
        # print("Creating seg from nonzero_mask with nonzero_label assignments")
        nonzero_mask = nonzero_mask.astype(int)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    # print(f"Exiting crop_to_nonzero with data.shape: {data.shape}, seg.shape: {seg.shape}, bbox: {bbox}")    
    return data, seg, bbox


def get_patient_identifiers_from_cropped_files(folder):
    # print(f"Entering get_patient_identifiers_from_cropped_files with folder: {folder}")
    ids = [i.split("/")[-1][:-4] for i in subfiles(folder, join=True, suffix=".npz")]
    # print(f"Found patient identifiers: {ids}")
    return ids


class ImageCropper(object):
    def __init__(self, num_threads, output_folder=None):
        """
        This one finds a mask of nonzero elements (must be nonzero in all modalities) and crops the image to that mask.
        In the case of BRaTS and ISLES data this results in a significant reduction in image size
        :param num_threads:
        :param output_folder: whete to store the cropped data
        :param list_of_files:
        """
        # print(f"Initializing ImageCropper with num_threads={num_threads}, output_folder={output_folder}")
        self.output_folder = output_folder
        self.num_threads = num_threads

        if self.output_folder is not None:
            # print(f"Ensuring output folder exists: {self.output_folder}")
            maybe_mkdir_p(self.output_folder)
            
        # print("ImageCropper initialized")    

    @staticmethod
    def crop(data, properties, seg=None):
        # print(f"Entering ImageCropper.crop with data.shape: {data.shape}, seg present: {seg is not None}")
        shape_before = data.shape
        # print(f"shape_before crop: {shape_before}")
        data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
        shape_after = data.shape
        # print(f"shape_after crop: {shape_after}")
        # print("before crop:", shape_before, "after crop:", shape_after, "spacing:",
            #   np.array(properties["original_spacing"]), "\n")

        properties["crop_bbox"] = bbox
        # print(f"Set properties['crop_bbox']: {bbox}")
        properties['classes'] = np.unique(seg)
        # print(f"Set properties['classes']: {properties['classes']}")
        seg[seg < -1] = 0
        properties["size_after_cropping"] = data[0].shape
        # print(f"Set properties['size_after_cropping']: {properties['size_after_cropping']}")
        return data, seg, properties

    @staticmethod
    def crop_from_list_of_files(data_files, seg_file=None):
        # print(f"Entering crop_from_list_of_files with data_files: {data_files}, seg_file: {seg_file}")
        data, seg, properties = load_case_from_list_of_files(data_files, seg_file)
        return ImageCropper.crop(data, properties, seg)

    def load_crop_save(self, case, case_identifier, overwrite_existing=False):
        # print(f"Entering load_crop_save for case_identifier: {case_identifier}, overwrite_existing: {overwrite_existing}")
        try:
            # print(f"Case files: {case}")
            npz_path = os.path.join(self.output_folder, f"{case_identifier}.npz")
            pkl_path = os.path.join(self.output_folder, f"{case_identifier}.pkl")
            # print(f"npz_path: {npz_path}, pkl_path: {pkl_path}")
            if overwrite_existing or not (isfile(npz_path) and isfile(pkl_path)):
                # print("Proceeding with cropping and saving")
                data, seg, properties = self.crop_from_list_of_files(case[:-1], case[-1])

                all_data = np.vstack((data, seg))
                np.savez_compressed(npz_path, data=all_data)
                # print(f"Saved compressed npz to {npz_path}")
                with open(pkl_path, 'wb') as f:
                    # print(f"Opening {pkl_path} for pickle dump")
                    pickle.dump(properties, f)
                    # print(f"Saved properties to {pkl_path}")
            else:
                print("Skipping save, files exist and overwrite_existing is False")        
        except Exception as e:
            # print("Exception in", case_identifier, ":")
            # print(e)
            raise e

    def get_list_of_cropped_files(self):
        # print(f"Listing cropped files in {self.output_folder}")
        files = subfiles(self.output_folder, join=True, suffix=".npz")
        # print(f"Found cropped files: {files}")
        return files

    def get_patient_identifiers_from_cropped_files(self):
        # print("Getting patient identifiers from cropped files via method")
        ids = [i.split("/")[-1][:-4] for i in self.get_list_of_cropped_files()]
        # print(f"Patient identifiers: {ids}")
        return ids


    def run_cropping(self, list_of_files, overwrite_existing=False, output_folder=None):
        """
        also copied ground truth nifti segmentation into the preprocessed folder so that we can use them for evaluation
        on the cluster
        :param list_of_files: list of list of files [[PATIENTID_TIMESTEP_0000.nii.gz], [PATIENTID_TIMESTEP_0000.nii.gz]]
        :param overwrite_existing:
        :param output_folder:
        :return:
        """
        # print(f"Entering run_cropping with {len(list_of_files)} cases, overwrite_existing: {overwrite_existing}, output_folder: {output_folder}")
        if output_folder is not None:
            self.output_folder = output_folder
            print(f"Updated output_folder: {self.output_folder}")

        output_folder_gt = os.path.join(self.output_folder, "gt_segmentations")
        # print(f"Ensuring gt_segmentations directory exists: {output_folder_gt}")
        maybe_mkdir_p(output_folder_gt)
        for j, case in enumerate(list_of_files):
            # print(f"Copying ground truth for case {j}")
            if case[-1] is not None:
                shutil.copy(case[-1], output_folder_gt)
                # print(f"Copied {case[-1]} to {output_folder_gt}")

        list_of_args = []
        # print("Preparing arguments for multiprocessing")
        for case in list_of_files:
            case_id = get_case_identifier(case)
            # print(f"Prepared args for {case_id}")
            list_of_args.append((case, case_id, overwrite_existing))

        # print(f"Starting Pool with num_threads: {self.num_threads}")
        p = Pool(self.num_threads)
        p.starmap(self.load_crop_save, list_of_args)
        p.close()
        p.join()
        print("Completed run_cropping")

    def load_properties(self, case_identifier):
        pkl_path = os.path.join(self.output_folder, f"{case_identifier}.pkl")
        # print(f"Loading properties from {pkl_path}")
        with open(pkl_path, 'rb') as f:
            properties = pickle.load(f)
            # print(f"Loaded properties keys: {list(properties.keys())}")
        return properties

    def save_properties(self, case_identifier, properties):
        pkl_path = os.path.join(self.output_folder, f"{case_identifier}.pkl")
        # print(f"Saving properties to {pkl_path}")
        with open(pkl_path, 'wb') as f:
            pickle.dump(properties, f)
            # print(f"Saved properties for {case_identifier}")

