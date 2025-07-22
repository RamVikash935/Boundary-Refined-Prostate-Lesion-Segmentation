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

from multiprocessing.pool import Pool
from time import sleep

import matplotlib
from postprocessing_connected_components import determine_postprocessing
from training_default_data_augmentation import get_default_augmentation
from training_dataloading_dataset_loading import DataLoader3D, unpack_dataset
from evaluation_evaluator import aggregate_scores
from training_network_nnUNetTrainer import nnUNetTrainer
from network_architectures_neural_network import SegmentationNetwork
from paths import network_training_output_dir
from inference_segmentation_export import save_segmentation_nifti_from_softmax
from batchgenerators.utilities_file_and_folder_operations import *
import numpy as np
from utilities import to_one_hot
import shutil

matplotlib.use("agg")


class nnUNetTrainerCascadeFullRes(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, previous_trainer="nnUNetTrainer", fp16=False):
        print(f"DEBUG: Entering __init__ with plans_file={plans_file}, fold={fold}, output_folder={output_folder}, "
              f"dataset_directory={dataset_directory}, batch_dice={batch_dice}, stage={stage}, "
              f"unpack_data={unpack_data}, deterministic={deterministic}, previous_trainer={previous_trainer}, fp16={fp16}")
        super(nnUNetTrainerCascadeFullRes, self).__init__(
            plans_file, fold, output_folder, dataset_directory,
            batch_dice, stage, unpack_data, deterministic, fp16)
        print("DEBUG: Called super().__init__ for CascadeFullRes")
        self.init_args = (plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                          unpack_data, deterministic, previous_trainer, fp16)
        print(f"DEBUG: Set self.init_args = {self.init_args}")

        if self.output_folder is not None:
            print(f"DEBUG: output_folder is not None: {self.output_folder}")
            task = self.output_folder.split("/")[-3]
            print(f"DEBUG: Parsed task = {task}")
            plans_identifier = self.output_folder.split("/")[-2].split("__")[-1]
            print(f"DEBUG: Parsed plans_identifier = {plans_identifier}")

            folder_with_segs_prev_stage = join(
                network_training_output_dir, "3d_lowres",
                task, previous_trainer + "__" + plans_identifier, "pred_next_stage")
            print(f"DEBUG: Computed folder_with_segs_prev_stage = {folder_with_segs_prev_stage}")
            if not isdir(folder_with_segs_prev_stage):
                print("DEBUG: Missing folder for previous stage, raising RuntimeError")
                raise RuntimeError(
                    "Cannot run final stage of cascade. Run corresponding 3d_lowres first and predict the segmentations for the next stage"
                )
            self.folder_with_segs_from_prev_stage = folder_with_segs_prev_stage
            print(f"DEBUG: Set self.folder_with_segs_from_prev_stage = {self.folder_with_segs_from_prev_stage}")
        else:
            self.folder_with_segs_from_prev_stage = None
            print("DEBUG: output_folder is None, set folder_with_segs_from_prev_stage = None")

    def do_split(self):
        print("DEBUG: Entering do_split for CascadeFullRes")
        super(nnUNetTrainerCascadeFullRes, self).do_split()
        print("DEBUG: Called super().do_split()")
        for k in self.dataset:
            path = join(self.folder_with_segs_from_prev_stage, k + "_segFromPrevStage.npz")
            self.dataset[k]['seg_from_prev_stage_file'] = path
            print(f"DEBUG: Set dataset[{k}]['seg_from_prev_stage_file'] = {path}")
            assert isfile(path), f"seg from prev stage missing: {path}"
        for k in self.dataset_val:
            path = join(self.folder_with_segs_from_prev_stage, k + "_segFromPrevStage.npz")
            self.dataset_val[k]['seg_from_prev_stage_file'] = path
            print(f"DEBUG: Set dataset_val[{k}]['seg_from_prev_stage_file'] = {path}")
        for k in self.dataset_tr:
            path = join(self.folder_with_segs_from_prev_stage, k + "_segFromPrevStage.npz")
            self.dataset_tr[k]['seg_from_prev_stage_file'] = path
            print(f"DEBUG: Set dataset_tr[{k}]['seg_from_prev_stage_file'] = {path}")

    def get_basic_generators(self):
        print("DEBUG: Entering get_basic_generators for CascadeFullRes")
        self.load_dataset()
        print("DEBUG: Called load_dataset()")
        self.do_split()
        print("DEBUG: Called do_split()")
        if self.threeD:
            print("DEBUG: Creating 3D DataLoaders for CascadeFullRes")
            dl_tr = DataLoader3D(
                self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                True, oversample_foreground_percent=self.oversample_foreground_percent
            )
            print(f"DEBUG: Created dl_tr = {dl_tr}")
            dl_val = DataLoader3D(
                self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                True, oversample_foreground_percent=self.oversample_foreground_percent
            )
            print(f"DEBUG: Created dl_val = {dl_val}")
        else:
            print("DEBUG: Non-3D mode not supported, raising NotImplementedError")
            raise NotImplementedError
        return dl_tr, dl_val

    def process_plans(self, plans):
        print(f"DEBUG: Entering process_plans for CascadeFullRes with plans keys: {list(plans.keys())}")
        super(nnUNetTrainerCascadeFullRes, self).process_plans(plans)
        print("DEBUG: Called super().process_plans()")
        old_channels = self.num_input_channels
        self.num_input_channels += (self.num_classes - 1)
        print(f"DEBUG: Increased num_input_channels from {old_channels} to {self.num_input_channels} (adding previous stage seg channels)")


    def setup_DA_params(self):
        print("DEBUG: Entering setup_DA_params for CascadeFullRes")
        super().setup_DA_params()
        print("DEBUG: Called super().setup_DA_params()")
        self.data_aug_params['move_last_seg_chanel_to_data'] = True
        print("DEBUG: Set move_last_seg_chanel_to_data = True")
        self.data_aug_params['cascade_do_cascade_augmentations'] = True
        print("DEBUG: Set cascade_do_cascade_augmentations = True")

        self.data_aug_params['cascade_random_binary_transform_p'] = 0.4
        print("DEBUG: Set cascade_random_binary_transform_p = 0.4")
        self.data_aug_params['cascade_random_binary_transform_p_per_label'] = 1
        print("DEBUG: Set cascade_random_binary_transform_p_per_label = 1")
        self.data_aug_params['cascade_random_binary_transform_size'] = (1, 8)
        print("DEBUG: Set cascade_random_binary_transform_size = (1, 8)")

        self.data_aug_params['cascade_remove_conn_comp_p'] = 0.2
        print("DEBUG: Set cascade_remove_conn_comp_p = 0.2")
        self.data_aug_params['cascade_remove_conn_comp_max_size_percent_threshold'] = 0.15
        print("DEBUG: Set cascade_remove_conn_comp_max_size_percent_threshold = 0.15")
        self.data_aug_params['cascade_remove_conn_comp_fill_with_other_class_p'] = 0.0
        print("DEBUG: Set cascade_remove_conn_comp_fill_with_other_class_p = 0.0")

        # we have 2 channels now because the segmentation from the previous stage is stored in 'seg' as well until it
        # is moved to 'data' at the end
        self.data_aug_params['selected_seg_channels'] = [0, 1]
        print("DEBUG: Set selected_seg_channels = [0, 1]")
        # needed for converting the segmentation from the previous stage to one hot
        self.data_aug_params['all_segmentation_labels'] = list(range(1, self.num_classes))
        print(f"DEBUG: Set all_segmentation_labels = {self.data_aug_params['all_segmentation_labels']}")

    def initialize(self, training=True, force_load_plans=False):
        """
        For prediction of test cases just set training=False, this will prevent loading of training data and
        training batchgenerator initialization
        :param training:
        :return:
        """
        print(f"DEBUG: Entering initialize for CascadeFullRes(training={training}, force_load_plans={force_load_plans})")
        if force_load_plans or (self.plans is None):
            print("DEBUG: Loading plans for CascadeFullRes")
            self.load_plans_file()
            print("DEBUG: Loaded plans file")

        print("DEBUG: Processing plans for CascadeFullRes")
        self.process_plans(self.plans)
        print("DEBUG: Processed plans")

        print("DEBUG: Setting up DA params for CascadeFullRes")
        self.setup_DA_params()
        print("DEBUG: Setup DA params")

        self.folder_with_preprocessed_data = join(
            self.dataset_directory,
            self.plans['data_identifier'] + f"_stage{self.stage}"
        )
        print(f"DEBUG: Set folder_with_preprocessed_data = {self.folder_with_preprocessed_data}")

        if training:
            print("DEBUG: Training=True, re-running setup_DA_params")
            self.setup_DA_params()
            print("DEBUG: Setup DA params again")

            if self.folder_with_preprocessed_data is not None:
                print("DEBUG: Initializing basic generators")
                self.dl_tr, self.dl_val = self.get_basic_generators()
                print(f"DEBUG: Obtained dl_tr={self.dl_tr}, dl_val={self.dl_val}")

                if self.unpack_data:
                    print("DEBUG: unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("DEBUG: done unpacking dataset")
                else:
                    print("DEBUG: Skipping unpack, may be slow for 2D data")

                self.tr_gen, self.val_gen = get_default_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params['patch_size_for_spatialtransform'],
                    self.data_aug_params
                )
                print(f"DEBUG: Created tr_gen={self.tr_gen}, val_gen={self.val_gen}")
                self.print_to_log_file("TRAINING KEYS:\n %s" % str(self.dataset_tr.keys()))
                print("DEBUG: Logged training dataset keys")
                self.print_to_log_file("VALIDATION KEYS:\n %s" % str(self.dataset_val.keys()))
                print("DEBUG: Logged validation dataset keys")
        else:
            print("DEBUG: Training=False, skipping data generator initialization")

        print("DEBUG: Initializing network for CascadeFullRes")
        self.initialize_network()
        print("DEBUG: Initialized network")
        assert isinstance(self.network, SegmentationNetwork)
        print("DEBUG: Network is instance of SegmentationNetwork")
        self.was_initialized = True
        print(f"DEBUG: Set was_initialized = {self.was_initialized}")


    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):

        print(f"DEBUG: Entering validate(do_mirroring={do_mirroring}, "
            f"use_sliding_window={use_sliding_window}, step_size={step_size}, "
            f"save_softmax={save_softmax}, use_gaussian={use_gaussian}, overwrite={overwrite}, "
            f"validation_folder_name={validation_folder_name}, debug={debug}, all_in_gpu={all_in_gpu}, "
            f"segmentation_export_kwargs={segmentation_export_kwargs}, "
            f"run_postprocessing_on_folds={run_postprocessing_on_folds})")

        print("Entering validate method of nnunetTrainerCascadeFullRes")
        current_mode = self.network.training
        print(f"DEBUG: Saved current network training mode = {current_mode}")
        self.network.eval()
        print("DEBUG: Switched network to eval()")

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        print("DEBUG: Checked was_initialized")
        if self.dataset_val is None:
            print("DEBUG: dataset_val is None, loading dataset and splitting")
            self.load_dataset()
            self.do_split()
            print(f"DEBUG: Completed load and split: train={len(self.dataset_tr)}, val={len(self.dataset_val)}")

        # Determine export params
        if segmentation_export_kwargs is None:
            print("DEBUG: No segmentation_export_kwargs, using plans if available")
            if 'segmentation_export_params' in self.plans.keys():
                print("Using segmentation export parameters from plans")
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
                print("DEBUG: Retrieved export params from plans")
            else:
                print("Using default export parameters")
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            print("Using provided segmentation export parameters")
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']
            print("DEBUG: Retrieved export params from segmentation_export_kwargs")

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        print(f"Output folder: {output_folder}")
        maybe_mkdir_p(output_folder)
        print(f"DEBUG: Ensured directory exists: {output_folder}")

        if do_mirroring:
            mirror_axes = self.data_aug_params['mirror_axes']
            print(f"Mirroring enabled with axes: {mirror_axes}")
        else:
            mirror_axes = ()
            print("Mirroring disabled")

        pred_gt_tuples = []

        export_pool = Pool(2)
        results = []

        transpose_backward = self.plans.get('transpose_backward')

        # Loop through validation cases
        for k in self.dataset_val.keys():
            print(f"Processing validation case: {k}")
            properties = load_pickle(self.dataset[k]['properties_file'])
            data = np.load(self.dataset[k]['data_file'])['data']

            seg_from_prev_stage = np.load(join(self.folder_with_segs_from_prev_stage,
                                               k + "_segFromPrevStage.npz"))['data'][None]
            print(f"Data shape before concatenation: {data.shape}")

            data[-1][data[-1] == -1] = 0
            data_for_net = np.concatenate((data[:-1], to_one_hot(seg_from_prev_stage[0], range(1, self.num_classes))))

            softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data_for_net,
                                                                                 do_mirroring=do_mirroring,
                                                                                 mirror_axes=mirror_axes,
                                                                                 use_sliding_window=use_sliding_window,
                                                                                 step_size=step_size,
                                                                                 use_gaussian=use_gaussian,
                                                                                 all_in_gpu=all_in_gpu,
                                                                                 mixed_precision=self.fp16)[1]
            print(f"DEBUG: Retrieved softmax_pred shape = {softmax_pred.shape}")

            if transpose_backward is not None:
                print("Transposing softmax prediction back")
                transpose_backward = self.plans.get('transpose_backward')
                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in transpose_backward])
                print(f"DEBUG: Transposed softmax_pred with transpose_backward = {self.transpose_backward}")

            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            print(f"DEBUG: softmax_fname = {fname}")

            if save_softmax:
                softmax_fname = join(output_folder, fname + ".npz")
                print(f"Saving softmax prediction to: {softmax_fname}")
            else:
                softmax_fname = None
            """There is a problem with python process communication that prevents us from communicating objects 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""
            if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):
                print("Softmax prediction too large, saving temporarily as .npy")
                np.save(fname + ".npy", softmax_pred)
                softmax_pred = fname + ".npy"

            results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                     ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                       properties, interpolation_order, self.regions_class_order,
                                                       None, None,
                                                       softmax_fname, None, force_separate_z,
                                                       interpolation_order_z),
                                                      )
                                                     )
                           )
            print(f"DEBUG: Queued save_segmentation_nifti_from_softmax for {fname}")

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])
            print(f"DEBUG: Appended pred_gt tuple for {fname}")

        print("Waiting for export results...")
        _ = [i.get() for i in results]

        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        print("Aggregating validation scores")
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"), json_name=job_name,
                             json_author="Fabian", json_description="",
                             json_task=task)

        if run_postprocessing_on_folds:
            print("Running postprocessing on folds")
            self.print_to_log_file("determining postprocessing")
            determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug)

        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        print(f"Copying GT niftis to: {gt_nifti_folder}")
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError:
                    print(f"Failed copying {f}, attempt {attempts+1}")
                    attempts += 1
                    sleep(1)

        print("Restoring network training mode")
        self.network.train(current_mode)
        export_pool.close()
        export_pool.join()
        print("Validation complete")
