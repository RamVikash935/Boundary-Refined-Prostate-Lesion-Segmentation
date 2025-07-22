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


from paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier
from batchgenerators.utilities_file_and_folder_operations import *
from experiment_planning_summarize_plans import summarize_plans
from training_model_restore import recursive_find_python_class


def get_configuration_from_output_folder(folder):
    # split off network_training_output_dir
    print(f"[DEBUG] input folder: {folder}")
    folder = folder[len(network_training_output_dir):]
    print(f"[DEBUG] after stripping network_training_output_dir: {folder}")
    if folder.startswith("/"):
        print(f"[DEBUG] folder starts with '/'")
        folder = folder[1:]
        print(f"[DEBUG] after removing leading '/': {folder}")

    configuration, task, trainer_and_plans_identifier = folder.split("/")
    print(f"[DEBUG] configuration: {configuration}, task: {task}, trainer_and_plans_identifier: {trainer_and_plans_identifier}")
    trainer, plans_identifier = trainer_and_plans_identifier.split("__")
    print(f"[DEBUG] trainer: {trainer}, plans_identifier: {plans_identifier}")
    return configuration, task, trainer, plans_identifier


def get_default_configuration(network, task, network_trainer, plans_identifier=default_plans_identifier,
                              search_in='all_helper_fn_n_class',
                              base_module='all_helper_fn_n_class'):
    
    print(f"[DEBUG] network: {network}, task: {task}, network_trainer: {network_trainer}, plans_identifier: {plans_identifier}, search_in: {search_in}, base_module: {base_module}")
    assert network in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], "network can only be one of the following: '2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'"
    print(f"[DEBUG] network assertion passed")

    dataset_directory = join(preprocessing_output_dir, task)
    print(f"[DEBUG] dataset_directory: {dataset_directory}")

    if network == '2d':
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_2D.pkl")
        print(f"[DEBUG] selected 2D plans_file: {plans_file}")
    else:
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_3D.pkl")
        print(f"[DEBUG] selected 3D plans_file: {plans_file}")

    plans = load_pickle(plans_file)
    print(f"[DEBUG] loaded plans from {plans_file}")
    possible_stages = list(plans['plans_per_stage'].keys())
    print(f"[DEBUG] possible_stages: {possible_stages}")

    if (network == '3d_cascade_fullres' or network == "3d_lowres") and len(possible_stages) == 1:
        print(f"[DEBUG] invalid single stage for cascade or lowres: {possible_stages}")
        raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. This task does "
                           "not require the cascade. Run 3d_fullres instead")

    if network == '2d' or network == "3d_lowres":
        stage = 0
        print(f"[DEBUG] using stage: {stage}")
    else:
        stage = possible_stages[-1]
        print(f"[DEBUG] using last stage: {stage}")

    trainer_class = recursive_find_python_class(search_in, network_trainer,base_module)
    print(f"[DEBUG] trainer_class resolved: {trainer_class}")

    output_folder_name = join(network_training_output_dir, network, task, network_trainer + "__" + plans_identifier)
    print(f"[DEBUG] output_folder_name: {output_folder_name}")

    print("###############################################")
    print(f"[DEBUG] I am running the following nnUNet: {network}")
    print("My trainer class is: ", trainer_class)
    print("For that I will be using the following configuration:")
    summarize_plans(plans_file)
    print(f"[DEBUG] plans summary printed for {plans_file}")
    print(f"I am using stage {stage} from these plans")

    if (network == '2d' or len(possible_stages) > 1) and not network == '3d_lowres':
        batch_dice = True
        print("I am using batch dice + CE loss")
    else:
        batch_dice = False
        print("I am using sample dice + CE loss")
        
    print(f"[DEBUG] batch_dice: {batch_dice}")

    print("\nI am using data from this folder: ", join(dataset_directory, plans['data_identifier']))
    # print(f"[DEBUG] data folder: {join(dataset_directory, plans['data_identifier'])}")
    print("###############################################")

    return plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class

