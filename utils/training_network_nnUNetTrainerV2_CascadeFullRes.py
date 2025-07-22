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
from configuration import default_num_threads
from postprocessing_connected_components import determine_postprocessing
from training_data_augmentation_moreDA import get_moreDA_augmentation
from training_dataloading_dataset_loading import DataLoader3D, unpack_dataset
from evaluation_evaluator import aggregate_scores
from network_architectures_neural_network import SegmentationNetwork
from paths import network_training_output_dir
from inference_segmentation_export import save_segmentation_nifti_from_softmax
from batchgenerators.utilities_file_and_folder_operations import *
import numpy as np
from training_loss_functions_deep_supervision import MultipleOutputLoss2
from training_network_nnUNetTrainerV2 import nnUNetTrainerV2
from utilities import to_one_hot
import shutil

from torch import nn

matplotlib.use("agg")


class nnUNetTrainerV2CascadeFullRes(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True,
                 previous_trainer='nnUNetTrainerV2', fp16=False):
        # initialize parent class
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        print("Initializing nnUNetTrainerV2CascadeFullRes")
        print(f'[DEBUG] Called super().__init__ with plans_file={plans_file}, fold={fold}, '
              f'output_folder={output_folder}, dataset_directory={dataset_directory}, '
              f'batch_dice={batch_dice}, stage={stage}, unpack_data={unpack_data}, '
              f'deterministic={deterministic}, fp16={fp16}')

        # save init args for later
        self.init_args = (plans_file, fold, output_folder, dataset_directory,
                          batch_dice, stage, unpack_data,
                          deterministic, previous_trainer, fp16)
        print(f'[DEBUG] Set self.init_args = {self.init_args}')

        # determine previous stage segmentation folder
        if self.output_folder is not None:
            print(f'[DEBUG] output_folder is not None: {self.output_folder}')
            task = self.output_folder.split('/')[-3]
            print(f'[DEBUG] Parsed task = {task}')
            plans_identifier = self.output_folder.split('/')[-2].split('__')[-1]
            print(f'[DEBUG] Parsed plans_identifier = {plans_identifier}')

            folder_with_segs_prev_stage = join(
                network_training_output_dir,
                '3d_lowres',
                task,
                previous_trainer + '__' + plans_identifier,
                'pred_next_stage'
            )
            print(f'[DEBUG] Computed folder_with_segs_prev_stage = {folder_with_segs_prev_stage}')

            self.folder_with_segs_from_prev_stage = folder_with_segs_prev_stage
            print(f'[DEBUG] Set self.folder_with_segs_from_prev_stage = {self.folder_with_segs_from_prev_stage}')
            # Do not put segs_prev_stage into self.output_folder as we need to unpack them for performance and we
            # don't want to do that in self.output_folder because that one is located on some network drive.
        else:
            self.folder_with_segs_from_prev_stage = None
            print('[DEBUG] output_folder is None, set self.folder_with_segs_from_prev_stage = None')

    def do_split(self):
        print('[DEBUG] Entering do_split() method of nnUNetTrainerV2CascadeFullRes')
        super().do_split()
        print('[DEBUG] Called super().do_split()')
        for k in self.dataset:
            print(f'[DEBUG] Processing key in dataset: {k}')
            seg_file = join(self.folder_with_segs_from_prev_stage, k + '_segFromPrevStage.npz')
            self.dataset[k]['seg_from_prev_stage_file'] = seg_file
            print(f'[DEBUG] Set seg_from_prev_stage_file for dataset {k}: {seg_file}')
            assert isfile(seg_file), (
                'seg from prev stage missing: %s. '
                'Please run all 5 folds of the 3d_lowres configuration of this '
                'task!' % seg_file
            )
            print(f'[DEBUG] Verified existence of seg file: {seg_file}')
        for k in self.dataset_val:
            print(f'[DEBUG] Processing key in dataset_val: {k}')
            seg_file = join(self.folder_with_segs_from_prev_stage, k + '_segFromPrevStage.npz')
            self.dataset_val[k]['seg_from_prev_stage_file'] = seg_file
            print(f'[DEBUG] Set seg_from_prev_stage_file for dataset_val {k}: {seg_file}')
        for k in self.dataset_tr:
            print(f'[DEBUG] Processing key in dataset_tr: {k}')
            seg_file = join(self.folder_with_segs_from_prev_stage, k + '_segFromPrevStage.npz')
            self.dataset_tr[k]['seg_from_prev_stage_file'] = seg_file
            print(f'[DEBUG] Set seg_from_prev_stage_file for dataset_tr {k}: {seg_file}')

    def get_basic_generators(self):
        print('[DEBUG] Entering get_basic_generators() method of nnUNetTrainerV2CascadeFullRes')
        self.load_dataset()
        print('[DEBUG] Completed load_dataset()')
        self.do_split()
        print('[DEBUG] Completed do_split()')

        if self.threeD:
            print('[DEBUG] Using 3D DataLoader')
            dl_tr = DataLoader3D(
                self.dataset_tr,
                self.basic_generator_patch_size,
                self.patch_size,
                self.batch_size,
                True,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode='constant',
                pad_sides=self.pad_all_sides
            )
            print(f'[DEBUG] Created dl_tr: {dl_tr}')
            dl_val = DataLoader3D(
                self.dataset_val,
                self.patch_size,
                self.patch_size,
                self.batch_size,
                True,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode='constant',
                pad_sides=self.pad_all_sides
            )
            print(f'[DEBUG] Created dl_val: {dl_val}')
        else:
            print('[DEBUG] 2D mode not supported, raising error')
            raise NotImplementedError('2D has no cascade')

        print('[DEBUG] Returning dl_tr, dl_val')
        return dl_tr, dl_val

    def process_plans(self, plans):
        print('[DEBUG] Entering process_plans() method of nnUNetTrainerV2CascadeFullRes')
        super().process_plans(plans)
        print('[DEBUG] Called super().process_plans(plans)')
        self.num_input_channels += (self.num_classes - 1)  # for seg from prev stage
        print(f'[DEBUG] Updated self.num_input_channels to {self.num_input_channels}')

    def setup_DA_params(self):
        print('[DEBUG] Entering setup_DA_params()')
        super().setup_DA_params()
        print('[DEBUG] Called super().setup_DA_params()')

        self.data_aug_params['num_cached_per_thread'] = 2
        print(f"[DEBUG] Set data_aug_params['num_cached_per_thread'] = {self.data_aug_params['num_cached_per_thread']}")

        self.data_aug_params['move_last_seg_chanel_to_data'] = True
        print(f"[DEBUG] Set data_aug_params['move_last_seg_chanel_to_data'] = {self.data_aug_params['move_last_seg_chanel_to_data']}")

        self.data_aug_params['cascade_do_cascade_augmentations'] = True
        print(f"[DEBUG] Set data_aug_params['cascade_do_cascade_augmentations'] = {self.data_aug_params['cascade_do_cascade_augmentations']}")

        self.data_aug_params['cascade_random_binary_transform_p'] = 0.4
        print(f"[DEBUG] Set data_aug_params['cascade_random_binary_transform_p'] = {self.data_aug_params['cascade_random_binary_transform_p']}")
        self.data_aug_params['cascade_random_binary_transform_p_per_label'] = 1
        print(f"[DEBUG] Set data_aug_params['cascade_random_binary_transform_p_per_label'] = {self.data_aug_params['cascade_random_binary_transform_p_per_label']}")
        self.data_aug_params['cascade_random_binary_transform_size'] = (1, 8)
        print(f"[DEBUG] Set data_aug_params['cascade_random_binary_transform_size'] = {self.data_aug_params['cascade_random_binary_transform_size']}")

        self.data_aug_params['cascade_remove_conn_comp_p'] = 0.2
        print(f"[DEBUG] Set data_aug_params['cascade_remove_conn_comp_p'] = {self.data_aug_params['cascade_remove_conn_comp_p']}")
        self.data_aug_params['cascade_remove_conn_comp_max_size_percent_threshold'] = 0.15
        print(f"[DEBUG] Set data_aug_params['cascade_remove_conn_comp_max_size_percent_threshold'] = {self.data_aug_params['cascade_remove_conn_comp_max_size_percent_threshold']}")
        self.data_aug_params['cascade_remove_conn_comp_fill_with_other_class_p'] = 0.0
        print(f"[DEBUG] Set data_aug_params['cascade_remove_conn_comp_fill_with_other_class_p'] = {self.data_aug_params['cascade_remove_conn_comp_fill_with_other_class_p']}")

        # we have 2 channels now because the segmentation from the previous stage is stored in 'seg' as well until it
        # is moved to 'data' at the end
        self.data_aug_params['selected_seg_channels'] = [0, 1]
        print(f"[DEBUG] Set data_aug_params['selected_seg_channels'] = {self.data_aug_params['selected_seg_channels']}")
        # needed for converting the segmentation from the previous stage to one hot
        self.data_aug_params['all_segmentation_labels'] = list(range(1, self.num_classes))
        print(f"[DEBUG] Set data_aug_params['all_segmentation_labels'] = {self.data_aug_params['all_segmentation_labels']}")

    def initialize(self, training=True, force_load_plans=False):
        """
        For prediction of test cases just set training=False, this will prevent loading of training data and
        training batchgenerator initialization
        :param training:
        :return:
        """
        print(f'[DEBUG] Entering initialize method of nnUNetTrainerV2CascadeFullRes (training={training}, force_load_plans={force_load_plans})')
        if not self.was_initialized:
            print('[DEBUG] was_initialized is False')
            if force_load_plans or (self.plans is None):
                print('[DEBUG] Loading plans file')
                self.load_plans_file()
                print('[DEBUG] Completed load_plans_file()')

            self.process_plans(self.plans)
            print('[DEBUG] Completed process_plans()')

            self.setup_DA_params()
            print('[DEBUG] Completed setup_DA_params()')

            ################# Here we wrap the loss for deep supervision ############
            print('[DEBUG] Entering deep supervision loss setup')
            net_numpool = len(self.net_num_pool_op_kernel_sizes)
            print(f'[DEBUG] Computed net_numpool = {net_numpool}')

            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
            print(f'[DEBUG] Computed raw weights = {weights}')
           
            mask = np.array([True if i < net_numpool - 1 else False for i in range(net_numpool)])
            print(f'[DEBUG] Computed mask = {mask}')
            weights[~mask] = 0
            print(f'[DEBUG] Applied mask, zeroed lowest two weights, weights = {weights}')
            weights = weights / weights.sum()
            print(f'[DEBUG] Normalized weights = {weights}')
            self.ds_loss_weights = weights
            print(f'[DEBUG] Set ds_loss_weights = {self.ds_loss_weights}')
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            print('[DEBUG] Wrapped loss with MultipleOutputLoss2')
            ################# END ###################

            self.folder_with_preprocessed_data = join(
                self.dataset_directory,
                self.plans['data_identifier'] + f'_stage{self.stage}'
            )
            print(f"[DEBUG] Set folder_with_preprocessed_data = {self.folder_with_preprocessed_data}")

            if training:
                print('[DEBUG] training=True, setting up training generators and data unpacking')
                if not isdir(self.folder_with_segs_from_prev_stage):
                    print('[DEBUG] Missing previous stage segmentations, raising RuntimeError')
                    raise RuntimeError(
                        "Cannot run final stage of cascade. Run corresponding 3d_lowres first and predict the "
                        "segmentations for the next stage"
                    )
                self.dl_tr, self.dl_val = self.get_basic_generators()
                print('[DEBUG] Obtained dl_tr and dl_val from get_basic_generators()')
                if self.unpack_data:
                    print('[DEBUG] unpacking dataset')
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print('[DEBUG] Done unpacking dataset')
                else:
                    print('[DEBUG] INFO: Not unpacking data! Training may be slow due to that.')

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr,
                    self.dl_val,
                    self.data_aug_params['patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory
                )
                print('[DEBUG] Created tr_gen and val_gen via get_moreDA_augmentation()')
                self.print_to_log_file(f"TRAINING KEYS:\n {list(self.dataset_tr.keys())}",
                                       also_print_to_console=False)
                print('[DEBUG] Logged training keys')
                self.print_to_log_file(f"VALIDATION KEYS:\n {list(self.dataset_val.keys())}",
                                       also_print_to_console=False)
                print('[DEBUG] Logged validation keys')
            else:
                print('[DEBUG] training=False, skipping generator setup')

            self.initialize_network()
            print('[DEBUG] Initialized network')
            self.initialize_optimizer_and_scheduler()
            print('[DEBUG] Initialized optimizer and scheduler')

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
            print('[DEBUG] Network is instance of SegmentationNetwork or nn.DataParallel')
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
            print('[DEBUG] was_initialized is True, skipped initialization')

        self.was_initialized = True
        print('[DEBUG] Set was_initialized = True')


    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        print(f'[DEBUG] Entering validate of nnUNetTrainerV2CascadeFullRes (do_mirroring={do_mirroring}, use_sliding_window={use_sliding_window}, '
              f'step_size={step_size}, save_softmax={save_softmax}, use_gaussian={use_gaussian}, '
              f'overwrite={overwrite}, validation_folder_name={validation_folder_name}, debug={debug}, '
              f'all_in_gpu={all_in_gpu}, segmentation_export_kwargs={segmentation_export_kwargs}, '
              f'run_postprocessing_on_folds={run_postprocessing_on_folds})')
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        print('[DEBUG] Asserted was_initialized')

        current_mode = self.network.training
        print(f'[DEBUG] Saved current_mode = {current_mode}')
        self.network.eval()
        print('[DEBUG] Set network to eval mode')
        ds = self.network.do_ds
        print(f'[DEBUG] Saved deep supervision mode ds = {ds}')
        self.network.do_ds = False
        print('[DEBUG] Disabled deep supervision for validation')

        if segmentation_export_kwargs is None:
            print('[DEBUG] segmentation_export_kwargs is None')
            if 'segmentation_export_params' in self.plans.keys():
                print('[DEBUG] Found segmentation_export_params in plans')
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                print(f'[DEBUG] force_separate_z = {force_separate_z}')
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                print(f'[DEBUG] interpolation_order = {interpolation_order}')
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
                print(f'[DEBUG] interpolation_order_z = {interpolation_order_z}')
            else:
                print('[DEBUG] segmentation_export_params not in plans')
                force_separate_z = None
                print('[DEBUG] Set force_separate_z = None')
                interpolation_order = 1
                print('[DEBUG] Set interpolation_order = 1')
                interpolation_order_z = 0
                print('[DEBUG] Set interpolation_order_z = 0')
        else:
            print('[DEBUG] segmentation_export_kwargs provided')
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            print(f'[DEBUG] force_separate_z = {force_separate_z}')
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            print(f'[DEBUG] interpolation_order = {interpolation_order}')
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']
            print(f'[DEBUG] interpolation_order_z = {interpolation_order_z}')

        if self.dataset_val is None:
            print('[DEBUG] dataset_val is None, loading dataset and splitting')
            self.load_dataset()
            print('[DEBUG] Completed load_dataset()')
            self.do_split()
            print('[DEBUG] Completed do_split()')

        output_folder = join(self.output_folder, validation_folder_name)
        print(f'[DEBUG] Set output_folder = {output_folder}')
        maybe_mkdir_p(output_folder)
        print('[DEBUG] Ensured output_folder exists')
        my_input_args = {
            'do_mirroring': do_mirroring,
            'use_sliding_window': use_sliding_window,
            'step': step_size,
            'save_softmax': save_softmax,
            'use_gaussian': use_gaussian,
            'overwrite': overwrite,
            'validation_folder_name': validation_folder_name,
            'debug': debug,
            'all_in_gpu': all_in_gpu,
            'segmentation_export_kwargs': segmentation_export_kwargs,
        }
        print(f'[DEBUG] Prepared my_input_args = {my_input_args}')
        save_json(my_input_args, join(output_folder, "validation_args.json"))
        print('[DEBUG] Saved validation_args.json')

        if do_mirroring:
            print('[DEBUG] do_mirroring=True')
            if not self.data_aug_params['do_mirror']:
                print('[DEBUG] data_aug_params do_mirror is False, raising RuntimeError')
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
            print(f'[DEBUG] mirror_axes = {mirror_axes}')
        else:
            mirror_axes = ()
            print('[DEBUG] do_mirroring=False, set mirror_axes = ()')

        pred_gt_tuples = []
        print('[DEBUG] Initialized pred_gt_tuples list')
        export_pool = Pool(default_num_threads)
        print('[DEBUG] Created export_pool')
        results = []
        print('[DEBUG] Initialized results list')

        for k in self.dataset_val.keys():
            print(f'[DEBUG] Validating key = {k}')
            properties = load_pickle(self.dataset[k]['properties_file'])
            print(f'[DEBUG] Loaded properties for {k}')
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            print(f'[DEBUG] Parsed fname = {fname}')

            if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                    (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                print('[DEBUG] Need to (re)compute predictions for', fname)
                data = np.load(self.dataset[k]['data_file'])['data']
                print(f'[DEBUG] Loaded data with shape = {data.shape}')

                seg_from_prev_stage = np.load(join(self.folder_with_segs_from_prev_stage,
                                                   k + "_segFromPrevStage.npz"))['data'][None]
                print('[DEBUG] Loaded seg_from_prev_stage of shape =', seg_from_prev_stage.shape)

                print(k, data.shape)
                data[-1][data[-1] == -1] = 0
                print('[DEBUG] Replaced -1 in last channel with 0')

                data_for_net = np.concatenate((data[:-1], to_one_hot(seg_from_prev_stage[0], range(1, self.num_classes))))
                print('[DEBUG] Created data_for_net with shape =', data_for_net.shape)

                softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(
                    data_for_net,
                    do_mirroring=do_mirroring,
                    mirror_axes=mirror_axes,
                    use_sliding_window=use_sliding_window,
                    step_size=step_size,
                    use_gaussian=use_gaussian,
                    all_in_gpu=all_in_gpu,
                    mixed_precision=self.fp16
                )[1]
                print('[DEBUG] Obtained softmax_pred with shape =', softmax_pred.shape)

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])
                print('[DEBUG] Transposed softmax_pred to shape =', softmax_pred.shape)

                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                else:
                    softmax_fname = None
                print(f'[DEBUG] Set softmax_fname = {softmax_fname}')

                """There is a problem with python process communication that prevents us from communicating objects 
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
                communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
                filename or np.ndarray and will handle this automatically"""
                if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):
                    np.save(join(output_folder, fname + ".npy"), softmax_pred)
                    print('[DEBUG] Saved large softmax_pred to .npy file')
                    softmax_pred = join(output_folder, fname + ".npy")
                    print(f'[DEBUG] Set softmax_pred path = {softmax_pred}')

                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, interpolation_order, None, None, None,
                                                           softmax_fname, None, force_separate_z,
                                                           interpolation_order_z),)))
                print('[DEBUG] Appended async save task to results')

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])
            print(f'[DEBUG] Added pred_gt_tuple for {fname}')

        _ = [i.get() for i in results]
        print('[DEBUG] Completed all export_pool tasks')
        self.print_to_log_file("finished prediction")
        print('[DEBUG] Logged finished prediction')

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        print('[DEBUG] Logged evaluation of raw predictions')
        task = self.dataset_directory.split("/")[-1]
        print(f'[DEBUG] Parsed task = {task}')
        job_name = self.experiment_name
        print(f'[DEBUG] Set job_name = {job_name}')
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name + f" val tiled {use_sliding_window}",
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads)
        print('[DEBUG] Completed aggregate_scores')

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well            
            self.print_to_log_file("determining postprocessing")
            print('[DEBUG] Logged determining postprocessing')
            determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
            print('[DEBUG] Completed determine_postprocessing')
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!            

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        print(f'[DEBUG] Set gt_nifti_folder = {gt_nifti_folder}')
        maybe_mkdir_p(gt_nifti_folder)
        print('[DEBUG] Ensured gt_nifti_folder exists')
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            print(f'[DEBUG] Copying ground truth nifti file = {f} to {gt_nifti_folder}')
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                print(f'[DEBUG] Attempt {attempts+1} to copy {f}')
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                    print(f'[DEBUG] Successfully copied {f}')
                except OSError as ex:
                    e = ex
                    attempts += 1
                    print(f'[DEBUG] Copy failed with OSError: {e}, retrying...')
                    sleep(1)
            if not success:
                print(f"[DEBUG] Could not copy gt nifti file {f} into folder {gt_nifti_folder}")
                if e is not None:
                    raise e

        # restore network deep supervision mode
        self.network.train(current_mode)
        print(f'[DEBUG] Restored network training mode to {current_mode}')
        self.network.do_ds = ds
        print(f'[DEBUG] Restored deep supervision mode ds to {ds}')

