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

sys.path.append(os.path.abspath('/home/user/ram/main_file/all_helper_fn_n_class'))


import torch
import torch._dynamo

# Suppress Triton-related errors and fallback to eager mode
torch._dynamo.config.suppress_errors = True

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignores only UserWarnings

import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)  # Hides TorchDynamo warnings

from all_helper_fn_n_class.batchgenerators.utilities_file_and_folder_operations import *
from all_helper_fn_n_class.configuration import default_num_threads
from all_helper_fn_n_class.experiment_planning_utils import split_4d
from all_helper_fn_n_class.utilities import remove_trailing_slash

def crawl_and_remove_hidden_from_decathlon(folder):
    folder = remove_trailing_slash(folder)
    assert folder.split('/')[-1].startswith("Task"), "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"
    subf = subfolders(folder, join=False)
    assert 'imagesTr' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"
    assert 'imagesTs' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"
    assert 'labelsTr' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"
    _ = [os.remove(i) for i in subfiles(folder, prefix=".")]
    _ = [os.remove(i) for i in subfiles(join(folder, 'imagesTr'), prefix=".")]
    _ = [os.remove(i) for i in subfiles(join(folder, 'labelsTr'), prefix=".")]
    _ = [os.remove(i) for i in subfiles(join(folder, 'imagesTs'), prefix=".")]

def split_4d_decathlon_entry(input_folder, num_processes=default_num_threads, output_task_id=None):
    """
    Converts 4D NIfTI MSD data to nnU-Net format with separate 3D NIfTIs for each modality.

    Args:
        input_folder (str): Path to TaskXX_TASKNAME folder as downloaded from MSD.
        num_processes (int, optional): Number of processes to use. Defaults to default_num_threads.
        output_task_id (int, optional): If specified, overwrites the task ID in the output. Defaults to None.
    """
    crawl_and_remove_hidden_from_decathlon(input_folder)
    split_4d(input_folder, num_processes, output_task_id)

split_4d_decathlon_entry('/home/user/ram/raw/Task05_Prostate',8,)



