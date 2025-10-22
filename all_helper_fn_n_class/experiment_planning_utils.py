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
    json_file = join(base_folder_splitted, "dataset.json")
    
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
        
    num_modalities = len(d['modality'].keys())
    
    for idx, tr in enumerate(training_files):
        cur_pat = []
        
        for mod in range(num_modalities):
            img_name = tr['image'].split("/")[-1][:-7] + "_%04.0d.nii.gz" % mod
            img_path = join(base_folder_splitted, "imagesTr", img_name)
            cur_pat.append(img_path)

        lbl_name = tr['label'].split("/")[-1]
        lbl_path = join(base_folder_splitted, "labelsTr", lbl_name)
        cur_pat.append(lbl_path)
        lists.append(cur_pat)
    
    modality_map = {int(i): d['modality'][i] for i in d['modality'].keys()}
    return lists, modality_map



def create_lists_from_splitted_dataset_folder(folder):
    """
    does not rely on dataset.json
    :param folder:
    :return:
    """
    caseIDs = get_caseIDs_from_splitted_dataset_folder(folder)
    list_of_lists = []
    
    for idx, f in enumerate(caseIDs):
        subfiles_list = subfiles(
            folder,
            prefix=f,
            suffix=".nii.gz",
            join=True,
            sort=True
        )
        list_of_lists.append(subfiles_list)
        
    return list_of_lists



def get_caseIDs_from_splitted_dataset_folder(folder):
    files = subfiles(folder, suffix=".nii.gz", join=False)
    # all files must be .nii.gz and have 4 digit modality index
    trimmed = [i[:-12] for i in files]
    
    # only unique patient ids
    unique_ids = np.unique(trimmed)

    return unique_ids


def crop(task_string, override=False, num_threads=default_num_threads):
    cropped_out_dir = join(nnUNet_cropped_data, task_string)
    
    maybe_mkdir_p(cropped_out_dir)

    if override and isdir(cropped_out_dir):
        shutil.rmtree(cropped_out_dir)
        maybe_mkdir_p(cropped_out_dir)
    
    splitted_4d_output_dir_task = join(nnUNet_raw_data, task_string)
    lists, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)
    imgcrop = ImageCropper(num_threads, cropped_out_dir)
    imgcrop.run_cropping(lists, overwrite_existing=override)
    src_json = join(nnUNet_raw_data, task_string, "dataset.json")
    dst_json = join(cropped_out_dir, "dataset.json")
    shutil.copy(src_json, cropped_out_dir)


def analyze_dataset(task_string, override=False, collect_intensityproperties=True, num_processes=default_num_threads):
    cropped_out_dir = join(nnUNet_cropped_data, task_string)
    dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=override, num_processes=num_processes)
    _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)


def plan_and_preprocess(task_string, processes_lowres=default_num_threads, processes_fullres=3, no_preprocessing=False):
    from experiment_planning_experiment_planner_baseline_2DUNet import ExperimentPlanner2D
    from experiment_planning_experiment_planner_baseline_3DUNet import ExperimentPlanner

    preprocessing_output_dir_this_task_train = join(preprocessing_output_dir, task_string)
    cropped_out_dir = join(nnUNet_cropped_data, task_string)

    maybe_mkdir_p(preprocessing_output_dir_this_task_train)
    
    src_pkl = join(cropped_out_dir, "dataset_properties.pkl")
    shutil.copy(src_pkl, preprocessing_output_dir_this_task_train)

    src_json = join(nnUNet_raw_data, task_string, "dataset.json")
    shutil.copy(src_json, preprocessing_output_dir_this_task_train)

    exp_planner = ExperimentPlanner(cropped_out_dir, preprocessing_output_dir_this_task_train)
    
    exp_planner.plan_experiment()
    
    if not no_preprocessing:
        exp_planner.run_preprocessing((processes_lowres, processes_fullres))
    
    exp_planner_2d = ExperimentPlanner2D(cropped_out_dir, preprocessing_output_dir_this_task_train)
    exp_planner_2d.plan_experiment()
    
    if not no_preprocessing:
        exp_planner_2d.run_preprocessing(processes_fullres)
        
    if not no_preprocessing:
        p = Pool(default_num_threads)
        stages = [
            i for i in subdirs(preprocessing_output_dir_this_task_train, join=True, sort=True)
            if "stage" in i.split("/")[-1]
        ]
    
        for s in stages:
            stage_name = s.split("/")[-1]
            list_of_npz_files = subfiles(s, True, None, ".npz", True)
            list_of_pkl_files = [i[:-4] + ".pkl" for i in list_of_npz_files]
            all_classes = []
            for pk in list_of_pkl_files:
                with open(pk, 'rb') as f:
                    props = pickle.load(f)
                all_classes_tmp = np.array(props['classes'])
                filtered = all_classes_tmp[all_classes_tmp >= 0]
                all_classes.append(filtered)
            args = zip(list_of_npz_files, list_of_pkl_files, all_classes)
            p.map(add_classes_in_slice_info, args)
            
        p.close()
        p.join()


def add_classes_in_slice_info(args):
    """
    We need this for 2D dataloader with oversampling. As of now it will detect slices that contain specific classes
    at run time, meaning it needs to iterate over an entire patient just to extract one slice. That is obviously bad,
    so we are doing this once beforehand and just give the dataloader the info it needs in the patients pkl file.
    """
    npz_file, pkl_file, all_classes = args
    seg_map = np.load(npz_file)['data'][-1]

    with open(pkl_file, 'rb') as f:
        props = pickle.load(f)
    classes_in_slice = OrderedDict()
    for axis in range(3):
        other_axes = tuple(i for i in range(3) if i != axis)
        classes_in_slice[axis] = OrderedDict()
        for c in all_classes:
            valid_slices = np.where(np.sum(seg_map == c, axis=other_axes) > 0)[0]
            classes_in_slice[axis][c] = valid_slices

    number_of_voxels_per_class = OrderedDict()
    for c in all_classes:
        count = int(np.sum(seg_map == c))
        number_of_voxels_per_class[c] = count
    
    props['classes_in_slice_per_axis'] = classes_in_slice
    props['number_of_voxels_per_class'] = number_of_voxels_per_class
    with open(pkl_file, 'wb') as f:
        pickle.dump(props, f)

