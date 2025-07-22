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
from batchgenerators.utilities_file_and_folder_operations import *
import argparse
from preprocessing_preprocessing import resample_data_or_seg
from batchgenerators.utilities_file_and_folder_operations import maybe_mkdir_p

from run_default_configuration import get_default_configuration
from multiprocessing import Pool

from training_model_restore import recursive_find_python_class
from training_network_nnUNetTrainer import nnUNetTrainer


def resample_and_save(predicted, target_shape, output_file, force_separate_z=False,
                      interpolation_order=1, interpolation_order_z=0):
    print(f"[DEBUG] resample_and_save called with output_file: {output_file}")
    if isinstance(predicted, str):
        print(f"[DEBUG] predicted is filepath: {predicted}")
        assert isfile(predicted), "If isinstance(segmentation_softmax, str) then isfile(segmentation_softmax) must be True"
        del_file = deepcopy(predicted)
        predicted = np.load(predicted)
        print(f"[DEBUG] loaded numpy array from {del_file}")
        os.remove(del_file)
        print(f"[DEBUG] removed temporary file {del_file}")

    predicted_new_shape = resample_data_or_seg(predicted, target_shape, False, order=interpolation_order,
                                               do_separate_z=force_separate_z, order_z=interpolation_order_z)
    print(f"[DEBUG] resampled data to new shape: {predicted_new_shape.shape}")
    seg_new_shape = predicted_new_shape.argmax(0)
    print(f"[DEBUG] computed argmax segmentation shape: {seg_new_shape.shape}")
    np.savez_compressed(output_file, data=seg_new_shape.astype(np.uint8))
    print(f"[DEBUG] saved compressed segmentation to {output_file}")


def predict_next_stage(trainer, stage_to_be_predicted_folder):
    print(f"[DEBUG] predict_next_stage called, output folder base: {stage_to_be_predicted_folder}")
    output_folder = join(pardir(trainer.output_folder), "pred_next_stage")
    print(f"[DEBUG] output_folder: {output_folder}")
    maybe_mkdir_p(output_folder)
    print(f"[DEBUG] ensured output_folder exists")

    if 'segmentation_export_params' in trainer.plans.keys():
        force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
        interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
        interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
        print(f"[DEBUG] using export params from plans: force_separate_z={force_separate_z}, interpolation_order={interpolationation_order}, interpolation_order_z={interpolation_order_z}")
    else:
        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0
        print(f"[DEBUG] using default export params: force_separate_z={force_separate_z}, interpolation_order={interpolation_order}, interpolation_order_z={interpolation_order_z}")

    export_pool = Pool(2)
    results = []
    print(f"[DEBUG] initialized multiprocessing pool and results list")

    for pat in trainer.dataset_val.keys():
        print(f"[DEBUG] processing patient: {pat}")
        data_file = trainer.dataset_val[pat]['data_file']
        print(f"[DEBUG] data_file: {data_file}")
        data_preprocessed = np.load(data_file)['data'][:-1]
        print(f"[DEBUG] loaded preprocessed data shape: {data_preprocessed.shape}")

        predicted_probabilities = trainer.predict_preprocessed_data_return_seg_and_softmax(
            data_preprocessed, do_mirroring=trainer.data_aug_params["do_mirror"],
            mirror_axes=trainer.data_aug_params['mirror_axes'], mixed_precision=trainer.fp16)[1]
        print(f"[DEBUG] obtained predicted_probabilities with shape: {predicted_probabilities.shape}")

        data_file_nofolder = data_file.split("/")[-1]
        data_file_nextstage = join(stage_to_be_predicted_folder, data_file_nofolder)
        print(f"[DEBUG] data_file_nextstage: {data_file_nextstage}")
        data_nextstage = np.load(data_file_nextstage)['data']
        target_shp = data_nextstage.shape[1:]
        print(f"[DEBUG] target_shp for resampling: {target_shp}")
        output_file = join(output_folder, data_file_nextstage.split("/")[-1][:-4] + "_segFromPrevStage.npz")
        print(f"[DEBUG] output_file: {output_file}")

        if np.prod(predicted_probabilities.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
            np.save(output_file[:-4] + ".npy", predicted_probabilities)
            print(f'[DEBUG] saved large predicted_probabilities to numpy file: {output_file[:-4] + ".npy"}')
            predicted_probabilities = output_file[:-4] + ".npy"
        
        results.append(export_pool.starmap_async(resample_and_save, [(predicted_probabilities, target_shp, output_file,
                                                                      force_separate_z, interpolation_order,
                                                                      interpolation_order_z)]))
        print(f"[DEBUG] appended async resample_and_save task for patient: {pat}")

    _ = [i.get() for i in results]
    print(f"[DEBUG] collected all async results")
    export_pool.close()
    print(f"[DEBUG] closed export_pool")
    export_pool.join()
    print(f"[DEBUG] joined export_pool")



if __name__ == "__main__":
    """
    RUNNING THIS SCRIPT MANUALLY IS USUALLY NOT NECESSARY. USE THE run_training.py FILE!
    
    This script is intended for predicting all the low resolution predictions of 3d_lowres for the next stage of the 
    cascade. It needs to run once for each fold so that the segmentation is only generated for the validation set 
    and not on the data the network was trained on. Run it with
    python predict_next_stage TRAINERCLASS TASK FOLD"""

    parser = argparse.ArgumentParser()
    parser.add_argument("network_trainer")
    parser.add_argument("task")
    parser.add_argument("fold", type=int)

    args = parser.parse_args()

    trainerclass = args.network_trainer
    task = args.task
    fold = args.fold

    plans_file, folder_with_preprocessed_data, output_folder_name, dataset_directory, batch_dice, stage = \
        get_default_configuration("3d_lowres", task)

    trainer_class = recursive_find_python_class("all_helper_fn_n_class",
                                                trainerclass,
                                                "all_helper_fn_n_class")

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")
    else:
        assert issubclass(trainer_class,
                          nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"

    trainer = trainer_class(plans_file, fold, folder_with_preprocessed_data, output_folder=output_folder_name,
                            dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage)

    trainer.initialize(False)
    trainer.load_dataset()
    trainer.do_split()
    trainer.load_best_checkpoint(train=False)

    stage_to_be_predicted_folder = join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1)
    output_folder = join(pardir(trainer.output_folder), "pred_next_stage")
    maybe_mkdir_p(output_folder)

    predict_next_stage(trainer, stage_to_be_predicted_folder)
