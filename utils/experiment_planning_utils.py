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

import json
import os
import pickle
import shutil
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
from batchgenerators.utilities_file_and_folder_operations import join, isdir, maybe_mkdir_p, subfiles, subdirs, isfile
from configuration import default_num_threads
from experiment_planning_DatasetAnalyzer import DatasetAnalyzer
from experiment_planning_common_utils import split_4d_nifti
from paths import nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir
from preprocessing_cropping import ImageCropper



def split_4d(input_folder, num_processes=default_num_threads, overwrite_task_output_id=None):
    
    assert isdir(join(input_folder, "imagesTr")) and isdir(join(input_folder, "labelsTr")) and isfile(join(input_folder, "dataset.json")), \
        "The input folder must be a valid Task folder from the Medical Segmentation Decathlon with at least the " "imagesTr and labelsTr subfolders and the dataset.json file"

    while input_folder.endswith("/"):
        input_folder = input_folder[:-1]

    full_task_name = input_folder.split("/")[-1]

    assert full_task_name.startswith("Task"), "The input folder must point to a folder that starts with TaskXX_"

    first_underscore = full_task_name.find("_")
    assert first_underscore == 6, "Input folder start with TaskXX with XX being a 3-digit id: 00, 01, 02 etc"

    input_task_id = int(full_task_name[4:6])
    
    if overwrite_task_output_id is None:
        overwrite_task_output_id = input_task_id

    task_name = full_task_name[7:]

    output_folder = join(nnUNet_raw_data, "Task%03.0d_" % overwrite_task_output_id + task_name)

    if isdir(output_folder):
        shutil.rmtree(output_folder)

    files = []
    output_dirs = []

    maybe_mkdir_p(output_folder)
    
    for subdir in ["imagesTr", "imagesTs"]:
        curr_out_dir = join(output_folder, subdir)
        if not isdir(curr_out_dir):
            os.mkdir(curr_out_dir)
        curr_dir = join(input_folder, subdir)
        nii_files = [join(curr_dir, i) for i in os.listdir(curr_dir) if i.endswith(".nii.gz")]
        nii_files.sort()
        for n in nii_files:
            files.append(n)
            output_dirs.append(curr_out_dir)

    shutil.copytree(join(input_folder, "labelsTr"), join(output_folder, "labelsTr"))

    p = Pool(num_processes)
    p.starmap(split_4d_nifti, zip(files, output_dirs))
    p.close()
    p.join()
    shutil.copy(join(input_folder, "dataset.json"), output_folder)


def create_lists_from_splitted_dataset(base_folder_splitted):
    lists = []
    # print(f"[DEBUG] Initialized empty lists: {lists}")

    json_file = join(base_folder_splitted, "dataset.json")
    # print(f"[DEBUG] Constructed json_file path: {json_file}")

    with open(json_file) as jsn:
        d = json.load(jsn)
        # print(f"[DEBUG] Loaded JSON content: keys={list(d.keys())}")

        training_files = d['training']
        # print(f"[DEBUG] Extracted training_files (count={len(training_files)}): {training_files}")

    num_modalities = len(d['modality'].keys())
    # print(f"[DEBUG] Number of modalities: {num_modalities}")

    for idx, tr in enumerate(training_files):
        # print(f"[DEBUG] Looping training_files[{idx}]: {tr}")
        cur_pat = []
        # print(f"[DEBUG]   Initialized cur_pat: {cur_pat}")

        for mod in range(num_modalities):
            img_name = tr['image'].split("/")[-1][:-7] + "_%04.0d.nii.gz" % mod
            img_path = join(base_folder_splitted, "imagesTr", img_name)
            cur_pat.append(img_path)
            # print(f"[DEBUG]   Added modality {mod} image path: {img_path}")

        lbl_name = tr['label'].split("/")[-1]
        lbl_path = join(base_folder_splitted, "labelsTr", lbl_name)
        cur_pat.append(lbl_path)
        # print(f"[DEBUG]   Added label path: {lbl_path}")

        lists.append(cur_pat)
        # print(f"[DEBUG] Appended cur_pat to lists (lists now length={len(lists)}): {cur_pat}")

    modality_map = {int(i): d['modality'][i] for i in d['modality'].keys()}
    print(f"[DEBUG] Constructed modality_map: {modality_map}")

    return lists, modality_map



def create_lists_from_splitted_dataset_folder(folder):
    """
    does not rely on dataset.json
    :param folder:
    :return:
    """
    print(f"[DEBUG] Input folder: {folder}")

    caseIDs = get_caseIDs_from_splitted_dataset_folder(folder)
    print(f"[DEBUG] Retrieved caseIDs ({len(caseIDs)}): {caseIDs}")

    list_of_lists = []
    print(f"[DEBUG] Initialized empty list_of_lists: {list_of_lists}")

    for idx, f in enumerate(caseIDs):
        print(f"[DEBUG] Looping caseIDs[{idx}]: {f}")

        subfiles_list = subfiles(
            folder,
            prefix=f,
            suffix=".nii.gz",
            join=True,
            sort=True
        )
        print(f"[DEBUG]   Retrieved subfiles_list ({len(subfiles_list)}): {subfiles_list}")

        list_of_lists.append(subfiles_list)
        # print(f"[DEBUG]   Appended to list_of_lists (current length={len(list_of_lists)}): {list_of_lists}")

    print(f"[DEBUG] Final list_of_lists: {list_of_lists}")
    return list_of_lists



def get_caseIDs_from_splitted_dataset_folder(folder):
    print(f"[DEBUG] Input folder: {folder}")

    files = subfiles(folder, suffix=".nii.gz", join=False)
    print(f"[DEBUG] Retrieved files ({len(files)}): {files}")

    # all files must be .nii.gz and have 4 digit modality index
    trimmed = [i[:-12] for i in files]
    print(f"[DEBUG] Trimmed filenames to remove modality index and extension ({len(trimmed)}): {trimmed}")

    # only unique patient ids
    unique_ids = np.unique(trimmed)
    print(f"[DEBUG] Unique case IDs ({len(unique_ids)}): {unique_ids}")

    return unique_ids


def crop(task_string, override=False, num_threads=default_num_threads):
    cropped_out_dir = join(nnUNet_cropped_data, task_string)
    # print(f"[DEBUG] Set cropped_out_dir: {cropped_out_dir}")

    maybe_mkdir_p(cropped_out_dir)
    # print(f"[DEBUG] Ensured directory exists: {cropped_out_dir}")

    if override and isdir(cropped_out_dir):
        print(f"[DEBUG] Override is True and directory exists; removing: {cropped_out_dir}")
        shutil.rmtree(cropped_out_dir)
        print(f"[DEBUG] Removed existing directory: {cropped_out_dir}")
        maybe_mkdir_p(cropped_out_dir)
        print(f"[DEBUG] Re-created directory: {cropped_out_dir}")

    splitted_4d_output_dir_task = join(nnUNet_raw_data, task_string)
    # print(f"[DEBUG] Set splitted_4d_output_dir_task: {splitted_4d_output_dir_task}")

    lists, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)
    # print(f"[DEBUG] Retrieved lists of file paths (count={len(lists)}): {lists}")

    imgcrop = ImageCropper(num_threads, cropped_out_dir)
    # print(f"[DEBUG] Initialized ImageCropper with num_threads={num_threads}, output_dir={cropped_out_dir}")

    imgcrop.run_cropping(lists, overwrite_existing=override)
    # print(f"[DEBUG] Ran cropping with overwrite_existing={override}")

    src_json = join(nnUNet_raw_data, task_string, "dataset.json")
    dst_json = join(cropped_out_dir, "dataset.json")
    shutil.copy(src_json, cropped_out_dir)
    # print(f"[DEBUG] Copied dataset.json to cropped_out_dir")



def analyze_dataset(task_string, override=False, collect_intensityproperties=True, num_processes=default_num_threads):
    print(f"[DEBUG] analyze_dataset called with task_string={task_string}, override={override}, collect_intensityproperties={collect_intensityproperties}, num_processes={num_processes}")

    cropped_out_dir = join(nnUNet_cropped_data, task_string)
    print(f"[DEBUG] Set cropped_out_dir: {cropped_out_dir}")

    dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=override, num_processes=num_processes)
    print(f"[DEBUG] Initialized DatasetAnalyzer with cropped_out_dir={cropped_out_dir}, overwrite={override}, num_processes={num_processes}")

    _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)
    print(f"[DEBUG] Ran analyze_dataset with collect_intensityproperties={collect_intensityproperties}")



def plan_and_preprocess(task_string, processes_lowres=default_num_threads, processes_fullres=3, no_preprocessing=False):
    from experiment_planning_experiment_planner_baseline_2DUNet import ExperimentPlanner2D
    from experiment_planning_experiment_planner_baseline_3DUNet import ExperimentPlanner
    print(f"[DEBUG] Called plan_and_preprocess with task_string={task_string}, processes_lowres={processes_lowres}, processes_fullres={processes_fullres}, no_preprocessing={no_preprocessing}")

    preprocessing_output_dir_this_task_train = join(preprocessing_output_dir, task_string)
    print(f"[DEBUG] Set preprocessing_output_dir_this_task_train: {preprocessing_output_dir_this_task_train}")

    cropped_out_dir = join(nnUNet_cropped_data, task_string)
    print(f"[DEBUG] Set cropped_out_dir: {cropped_out_dir}")

    maybe_mkdir_p(preprocessing_output_dir_this_task_train)
    print(f"[DEBUG] Ensured directory exists: {preprocessing_output_dir_this_task_train}")

    src_pkl = join(cropped_out_dir, "dataset_properties.pkl")
    print(f"[DEBUG] Copying dataset_properties.pkl from {src_pkl} to {preprocessing_output_dir_this_task_train}")
    shutil.copy(src_pkl, preprocessing_output_dir_this_task_train)
    print(f"[DEBUG] Copied dataset_properties.pkl")

    src_json = join(nnUNet_raw_data, task_string, "dataset.json")
    print(f"[DEBUG] Copying dataset.json from {src_json} to {preprocessing_output_dir_this_task_train}")
    shutil.copy(src_json, preprocessing_output_dir_this_task_train)
    print(f"[DEBUG] Copied dataset.json")

    exp_planner = ExperimentPlanner(cropped_out_dir, preprocessing_output_dir_this_task_train)
    print(f"[DEBUG] Initialized 3D ExperimentPlanner with cropped_out_dir={cropped_out_dir}, output_dir={preprocessing_output_dir_this_task_train}")

    exp_planner.plan_experiment()
    print(f"[DEBUG] Ran plan_experiment on 3D planner")

    if not no_preprocessing:
        print(f"[DEBUG] Running 3D preprocessing with processes_lowres={processes_lowres}, processes_fullres={processes_fullres}")
        exp_planner.run_preprocessing((processes_lowres, processes_fullres))
        print(f"[DEBUG] Completed 3D preprocessing")

    exp_planner_2d = ExperimentPlanner2D(cropped_out_dir, preprocessing_output_dir_this_task_train)
    print(f"[DEBUG] Initialized 2D ExperimentPlanner with cropped_out_dir={cropped_out_dir}, output_dir={preprocessing_output_dir_this_task_train}")

    exp_planner_2d.plan_experiment()
    print(f"[DEBUG] Ran plan_experiment on 2D planner")

    if not no_preprocessing:
        print(f"[DEBUG] Running 2D preprocessing with processes_fullres={processes_fullres}")
        exp_planner_2d.run_preprocessing(processes_fullres)
        print(f"[DEBUG] Completed 2D preprocessing")

    if not no_preprocessing:
        print(f"[DEBUG] Starting slice-wise class info aggregation")

        p = Pool(default_num_threads)
        print(f"[DEBUG] Created multiprocessing Pool with {default_num_threads} processes")

        stages = [
            i for i in subdirs(preprocessing_output_dir_this_task_train, join=True, sort=True)
            if "stage" in i.split("/")[-1]
        ]
        print(f"[DEBUG] Found stages directories: {stages}")

        for s in stages:
            stage_name = s.split("/")[-1]
            print(f"[DEBUG] Processing stage: {stage_name}")

            list_of_npz_files = subfiles(s, True, None, ".npz", True)
            print(f"[DEBUG] Retrieved .npz files ({len(list_of_npz_files)}): {list_of_npz_files}")

            list_of_pkl_files = [i[:-4] + ".pkl" for i in list_of_npz_files]
            print(f"[DEBUG] Corresponding .pkl files: {list_of_pkl_files}")

            all_classes = []
            for pk in list_of_pkl_files:
                print(f"[DEBUG] Loading props from {pk}")
                with open(pk, 'rb') as f:
                    props = pickle.load(f)
                all_classes_tmp = np.array(props['classes'])
                print(f"[DEBUG] Extracted classes array: {all_classes_tmp}")

                filtered = all_classes_tmp[all_classes_tmp >= 0]
                print(f"[DEBUG] Filtered non-negative classes: {filtered}")

                all_classes.append(filtered)
            print(f"[DEBUG] Collected all_classes for stage {stage_name}: {all_classes}")

            args = zip(list_of_npz_files, list_of_pkl_files, all_classes)
            print(f"[DEBUG] Mapping add_classes_in_slice_info over files")
            p.map(add_classes_in_slice_info, args)
            print(f"[DEBUG] Completed mapping for stage {stage_name}")

        p.close()
        print(f"[DEBUG] Closed multiprocessing Pool")

        p.join()
        print(f"[DEBUG] Joined multiprocessing Pool")




def add_classes_in_slice_info(args):
    """
    We need this for 2D dataloader with oversampling. As of now it will detect slices that contain specific classes
    at run time, meaning it needs to iterate over an entire patient just to extract one slice. That is obviously bad,
    so we are doing this once beforehand and just give the dataloader the info it needs in the patients pkl file.
    """
    print(f"[DEBUG] add_classes_in_slice_info called with args={args}")
    npz_file, pkl_file, all_classes = args
    print(f"[DEBUG] npz_file: {npz_file}")
    print(f"[DEBUG] pkl_file: {pkl_file}")
    print(f"[DEBUG] all_classes: {all_classes}")

    seg_map = np.load(npz_file)['data'][-1]
    print(f"[DEBUG] Loaded segmentation map from {npz_file}, shape: {seg_map.shape}")

    with open(pkl_file, 'rb') as f:
        props = pickle.load(f)
    print(f"[DEBUG] Loaded props from {pkl_file}: keys={list(props.keys())}")

    print(f"[DEBUG] Initializing classes_in_slice")
    classes_in_slice = OrderedDict()
    for axis in range(3):
        print(f"[DEBUG] Processing axis: {axis}")
        other_axes = tuple(i for i in range(3) if i != axis)
        print(f"[DEBUG] Other axes for sum: {other_axes}")

        classes_in_slice[axis] = OrderedDict()
        for c in all_classes:
            valid_slices = np.where(np.sum(seg_map == c, axis=other_axes) > 0)[0]
            print(f"[DEBUG] Found slices for class {c} on axis {axis}: {valid_slices}")

            classes_in_slice[axis][c] = valid_slices

    print(f"[DEBUG] Initializing number_of_voxels_per_class")
    number_of_voxels_per_class = OrderedDict()
    for c in all_classes:
        count = int(np.sum(seg_map == c))
        number_of_voxels_per_class[c] = count
        print(f"[DEBUG] Voxel count for class {c}: {count}")

    props['classes_in_slice_per_axis'] = classes_in_slice
    print(f"[DEBUG] Added 'classes_in_slice_per_axis' to props")

    props['number_of_voxels_per_class'] = number_of_voxels_per_class
    print(f"[DEBUG] Added 'number_of_voxels_per_class' to props")

    with open(pkl_file, 'wb') as f:
        pickle.dump(props, f)
    print(f"[DEBUG] Written updated props back to {pkl_file}")

