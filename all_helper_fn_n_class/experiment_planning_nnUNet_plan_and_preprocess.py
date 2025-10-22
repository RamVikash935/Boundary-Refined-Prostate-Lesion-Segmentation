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

from batchgenerators.utilities_file_and_folder_operations import *
from experiment_planning_DatasetAnalyzer import DatasetAnalyzer
from experiment_planning_utils import crop
from paths import *
import shutil
from utilities import convert_id_to_task_name
from preprocessing_sanity_checks import verify_dataset_integrity
from training_model_restore import recursive_find_python_class


def plan_and_preprocess(
    task_ids,
    planner3d_class_name='ExperimentPlanner3D_v21',
    planner2d_class_name='ExperimentPlanner2D_v21',
    skip_preprocessing=False,
    threads_low_res=8,
    threads_full_res=8,
    verify_dataset_integrity=False,
    overwrite_plans=None,
    overwrite_plans_identifier=None
):
    """
    Automated experiment planning and preprocessing for nnU-Net tasks.

    Parameters:
    - task_ids: list of int
        List of Task IDs to process (must match folders 'TaskXXX_' in raw data).
    - planner3d_class_name: str or None
        Name of the 3D experiment planner class (or None to skip 3D).
    - planner2d_class_name: str or None
        Name of the 2D experiment planner class (or None to skip 2D).
    - skip_preprocessing: bool
        If True, only run planning (no preprocessing).
    - threads_low_res: int
        Number of processes for low-res 3D preprocessing.
    - threads_full_res: int
        Number of processes for full-res 2D/3D preprocessing.
    - verify_dataset_integrity: bool
        If True, perform dataset integrity checks before processing.
    - overwrite_plans: str or None
        If set, path to an existing plans file to overwrite default settings.
    - overwrite_plans_identifier: str or None
        Unique identifier required when overwrite_plans is used.
    """    
    
    # Convert string 'None' to actual None
    if planner3d_class_name == 'None':
        planner3d_class_name = None
    if planner2d_class_name == 'None':
        planner2d_class_name = None

    # Overwrite logic validation
    if overwrite_plans is not None:
        if planner2d_class_name is not None:
            planner2d_class_name = None
        assert planner3d_class_name == 'ExperimentPlanner3D_v21_Pretrained', \
            "When using overwrite_plans you must set planner3d_class_name to 'ExperimentPlanner3D_v21_Pretrained'"
        
    # Prepare task names
    tasks = []
    for tid in task_ids:
        tid = int(tid)
        task_name = convert_id_to_task_name(tid)
    
        if verify_dataset_integrity:
            verify_dataset_integrity(join(nnUNet_raw_data, task_name))
        
        crop(task_name, False, threads_full_res)
        tasks.append(task_name)

    # Locate planner classes
    search_in = 'all_helper_fn_n_class'
    planner_3d = None
    planner_2d = None

    if planner3d_class_name is not None:
        planner_3d = recursive_find_python_class(
            [search_in], planner3d_class_name,
            current_module='nnunet.experiment_planning'
        )
        if planner_3d is None:
            raise RuntimeError(f"Could not find 3D planner class '{planner3d_class_name}'")
    if planner2d_class_name is not None:
        planner_2d = recursive_find_python_class(
            [search_in], planner2d_class_name,
            current_module='nnunet.experiment_planning'
        )
        if planner_2d is None:
            raise RuntimeError(f"Could not find 2D planner class '{planner2d_class_name}'")

    # Process each task
    for t in tasks:
        cropped_dir = join(nnUNet_cropped_data, t)
        prep_output_dir = join(preprocessing_output_dir, t)
    
        # Load modality info
        dataset_json = load_json(join(cropped_dir, 'dataset.json'))
        modalities = list(dataset_json['modality'].values())
        collect_intensity = any(m.lower() == 'ct' for m in modalities)
        
        # Analyze and fingerprint
        dataset_analyzer = DatasetAnalyzer(cropped_dir, overwrite=False, num_processes=threads_full_res)
        dataset_analyzer.analyze_dataset(collect_intensity)

        # Prepare output directories
        maybe_mkdir_p(prep_output_dir)
        shutil.copy(join(cropped_dir, 'dataset_properties.pkl'), prep_output_dir)
        shutil.copy(join(nnUNet_raw_data, t, 'dataset.json'), prep_output_dir)

        threads = (threads_low_res, threads_full_res)

        # Run 3D planning & preprocessing
        if planner_3d is not None:
            if overwrite_plans is not None:
                exp_planner = planner_3d(
                    cropped_dir, prep_output_dir,
                    overwrite_plans, overwrite_plans_identifier
                )
            else:
                exp_planner = planner_3d(cropped_dir, prep_output_dir)
            exp_planner.plan_experiment()
            if not skip_preprocessing:
                exp_planner.run_preprocessing(threads)

        # Run 2D planning & preprocessing
        if planner_2d is not None:
            exp_planner = planner_2d(cropped_dir, prep_output_dir)
            exp_planner.plan_experiment()
            if not skip_preprocessing:
                exp_planner.run_preprocessing(threads)



