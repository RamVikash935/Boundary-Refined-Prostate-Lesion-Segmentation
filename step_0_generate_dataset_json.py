
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


from typing import Tuple
import numpy as np
from all_helper_fn_n_class.utilities_file_and_folder_operations import *

def get_identifiers_from_splitted_files(folder: str):
    files = subfiles(folder, suffix='.nii.gz', join=False)
    base_ids = [i[:-7] for i in files]
    # base_ids = [i[:-12] for i in files]
    uniques = np.unique(base_ids)
    return uniques


def generate_dataset_json(output_file: str,
                          imagesTr_dir: str,
                          imagesTs_dir: str,
                          modalities: Tuple,
                          labels: dict,
                          dataset_name: str,
                          sort_keys=True,
                          license: str = "hands off!",
                          dataset_description: str = "",
                          dataset_reference: str = "",
                          dataset_release: str = '0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param sort_keys: In order to sort or not, the keys in dataset.json
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """    
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    # Build JSON dictionary
    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)

    json_dict['training'] = [
        {'image': f"./imagesTr/{i}.nii.gz", 'label': f"./labelsTr/{i}.nii.gz"}
        for i in train_identifiers
    ]

    json_dict['test'] = [f"./imagesTs/{i}.nii.gz" for i in test_identifiers]

    # Ensure correct output file path
    if os.path.isdir(output_file):
        output_file = os.path.join(output_file, 'dataset.json')
    elif not output_file.endswith("dataset.json"):

    save_json(json_dict, output_file, sort_keys=sort_keys)


output_file = '/home/user/ram/raw/Task05_Prostate/dataset.json'
imagesTr_dir = '/home/user/ram/raw/Task05_Prostate/imagesTr'
imagesTs_dir = '/home/user/ram/raw/Task05_Prostate/imagesTs'
modalities = ('T2',)
labels = {0:'background', 1:'tumor'}
dataset_name = 'PICAI Challenges'
sort_keys = True
license = "hands off!"
dataset_description = 'Prostate tumor segmentation'
dataset_reference = 'Charite Universittatsmedizin Berlin, German university'
dataset_release = '0.0'


generate_dataset_json(
    output_file=output_file,
    imagesTr_dir = imagesTr_dir,
    imagesTs_dir=imagesTs_dir,
    modalities=modalities,
    labels=labels,
    dataset_name=dataset_name,
    sort_keys=sort_keys,
    license=license,
    dataset_description=dataset_description,
    dataset_reference=dataset_reference,
    dataset_release=dataset_release
)

