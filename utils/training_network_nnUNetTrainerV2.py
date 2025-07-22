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

from collections import OrderedDict
from typing import Tuple

import torch
from training_data_augmentation_moreDA import get_moreDA_augmentation
from training_loss_functions_deep_supervision import MultipleOutputLoss2
from utilities import maybe_to_torch, to_cuda, softmax_helper
from network_architectures_generic_UNet import Generic_UNet
from network_architectures_initialization import InitWeights_He
from network_architectures_neural_network import SegmentationNetwork
from training_default_data_augmentation import default_2D_augmentation_params, get_patch_size, default_3D_augmentation_params
from training_dataloading_dataset_loading import unpack_dataset
from training_network_nnUNetTrainer import nnUNetTrainer
from sklearn.model_selection import KFold
from torch import nn
from torch.amp import autocast
from training_learning_rate_poly_lr import poly_lr
from batchgenerators.utilities_file_and_folder_operations import *


class nnUNetTrainerV2(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        
        print(f"[DEBUG] super().__init__ called with plans_file={plans_file}, fold={fold}, output_folder={output_folder}, dataset_directory={dataset_directory}, batch_dice={batch_dice}, stage={stage}, unpack_data={unpack_data}, deterministic={deterministic}, fp16={fp16}")
        self.max_num_epochs = 1000
        print(f"[DEBUG] self.max_num_epochs set in nnUNetTrainerV2 to {self.max_num_epochs}")
        self.initial_lr = 1e-2
        print(f"[DEBUG] self.initial_lr set in nnUNetTrainerV2 to {self.initial_lr}")
        self.deep_supervision_scales = None
        print(f"[DEBUG] self.deep_supervision_scales set in nnUNetTrainerV2 to {self.deep_supervision_scales}")
        self.ds_loss_weights = None
        print(f"[DEBUG] self.ds_loss_weights set in nnUNetTrainerV2 to {self.ds_loss_weights}")

        self.pin_memory = True
        print(f"[DEBUG] self.pin_memory set in nnUNetTrainerV2 to {self.pin_memory}")

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """        
        print(f"[DEBUG] initialize of nnUNetTrainerV2 called with training={training}, force_load_plans={force_load_plans}")
        if not self.was_initialized:
            print("[DEBUG] was_initialized is False, proceeding with initialization")
            maybe_mkdir_p(self.output_folder)
            print(f"[DEBUG] maybe_mkdir_p called on {self.output_folder}")

            if force_load_plans or (self.plans is None):
                print("[DEBUG] force_load_plans is True or self.plans is None, loading plans")
                self.load_plans_file()
                # print("[DEBUG] load_plans_file completed")

            # print("[DEBUG] processing plans")
            self.process_plans(self.plans)
            print(f"[DEBUG] process_plans completed with plans: {self.plans}")

            # print("[DEBUG] setting up data augmentation parameters")
            self.setup_DA_params()
            print(f"[DEBUG] setup_DA_params completed, data_aug_params: {self.data_aug_params}")

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            # print("[DEBUG] computing net_numpool from net_num_pool_op_kernel_sizes")
            net_numpool = len(self.net_num_pool_op_kernel_sizes)
            print(f"[DEBUG] net_numpool: {net_numpool}")

            print("[DEBUG] computing deep supervision weights")
            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss            
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
            print(f"[DEBUG] initial weights: {weights}")

            print("[DEBUG] applying mask to weights")
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            print(f"[DEBUG] mask: {mask}")
            weights[~mask] = 0
            print(f"[DEBUG] weights after mask: {weights}")
            weights = weights / weights.sum()
            print(f"[DEBUG] normalized weights: {weights}")
            self.ds_loss_weights = weights
            print(f"[DEBUG] self.ds_loss_weights set to {self.ds_loss_weights}")

            print("[DEBUG] wrapping loss with MultipleOutputLoss2")
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            print(f"[DEBUG] self.loss wrapped: {self.loss}")
            ################# END ###################

            self.folder_with_preprocessed_data = join(
                self.dataset_directory, self.plans['data_identifier'] + f"_stage{self.stage}")
            print(f"[DEBUG] folder_with_preprocessed_data set to {self.folder_with_preprocessed_data}")

            if training:
                print("[DEBUG] training is True, getting basic generators")
                self.dl_tr, self.dl_val = self.get_basic_generators()
                print(f"[DEBUG] obtained dl_tr: {self.dl_tr}, dl_val: {self.dl_val}")
                if self.unpack_data:
                    print("[DEBUG] unpack_data is True, unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    # print("[DEBUG] unpack_dataset completed")
                else:
                    print("[DEBUG] unpack_data is False, skipping unpack_dataset")
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                # print("[DEBUG] setting up data augmentation generators")
                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params['patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                print(f"[DEBUG] tr_gen: {self.tr_gen}, val_gen: {self.val_gen}")

                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                print(f"[DEBUG] printed training keys to log: {self.dataset_tr.keys()}")
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
                print(f"[DEBUG] printed validation keys to log: {self.dataset_val.keys()}")
            else:
                print("[DEBUG] training is False, skipping generator and augmentation setup")

            print("[DEBUG] initializing network")
            self.initialize_network()
            # print(f"[DEBUG] initialize_network completed, network: {self.network}")
            print("[DEBUG] initializing optimizer and scheduler")
            self.initialize_optimizer_and_scheduler()
            print("[DEBUG] initialize_optimizer_and_scheduler completed")

            # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel)), \
                # f"[DEBUG] Assertion failed: network is {type(self.network)}"
            # print("[DEBUG] network instance assertion passed")
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
            print("[DEBUG] was_initialized is True, skipping initialization")
            
        self.was_initialized = True
        print(f"[DEBUG] self.was_initialized set to {self.was_initialized}")



    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """        
        print("[DEBUG] initialize_network of nnUNetTrainerV2 called")
        
        if self.threeD:
            conv_op = nn.Conv3d
            # print(f"[DEBUG] conv_op set to {conv_op}")
            dropout_op = nn.Dropout3d
            # print(f"[DEBUG] dropout_op set to {dropout_op}")
            norm_op = nn.InstanceNorm3d
            # print(f"[DEBUG] norm_op set to {norm_op}")
        else:
            conv_op = nn.Conv2d
            # print(f"[DEBUG] conv_op set to {conv_op}")
            dropout_op = nn.Dropout2d
            # print(f"[DEBUG] dropout_op set to {dropout_op}")
            norm_op = nn.InstanceNorm2d
            # print(f"[DEBUG] norm_op set to {norm_op}")

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        print(f"[DEBUG] norm_op_kwargs set to {norm_op_kwargs}")
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        print(f"[DEBUG] dropout_op_kwargs set to {dropout_op_kwargs}")
        net_nonlin = nn.LeakyReLU
        print(f"[DEBUG] net_nonlin set to {net_nonlin}")
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        print(f"[DEBUG] net_nonlin_kwargs set to {net_nonlin_kwargs}")
        
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        
        print(f"[DEBUG] network instantiated: {self.network}")
        if torch.cuda.is_available():
            print("[DEBUG] CUDA is available, moving network to GPU")
            self.network.cuda()
            print("[DEBUG] network moved to GPU")
        else:
            print("[DEBUG] CUDA not available, skipping .cuda()")
        
        self.network.inference_apply_nonlin = softmax_helper
        print(f"[DEBUG] inference_apply_nonlin set to {softmax_helper}")

    def initialize_optimizer_and_scheduler(self):
        print("[DEBUG] initialize_optimizer_and_scheduler of nnUNetTrainerV2 is called")
        assert self.network is not None, "self.initialize_network must be called first"
        # print(f"[DEBUG] assertion passed: self.network is {type(self.network)}")
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        print(f"[DEBUG] optimizer created: {self.optimizer}")
        self.lr_scheduler = None
        print("[DEBUG] lr_scheduler set to None")

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """        
        # print(f"[DEBUG] run_online_evaluation of nnUNetTrainerV2 is called with output={output}, target={target}")
        target = target[0]
        # print(f"[DEBUG] target[0]: {target}")
        output = output[0]
        # print(f"[DEBUG] output[0]: {output}")
        result = super().run_online_evaluation(output, target)
        # print(f"[DEBUG] super().run_online_evaluation returned: {result}")
        return result

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        print(f"[DEBUG] validate of nnUNetTrainerV2 is called with do_mirroring={do_mirroring}, use_sliding_window={use_sliding_window}, step_size={step_size}, save_softmax={save_softmax}, use_gaussian={use_gaussian}, overwrite={overwrite}, validation_folder_name={validation_folder_name}, debug={debug}, all_in_gpu={all_in_gpu}, segmentation_export_kwargs={segmentation_export_kwargs}, run_postprocessing_on_folds={run_postprocessing_on_folds}")
        ds = self.network.do_ds
        print(f"[DEBUG] saved original do_ds: {ds}")
        self.network.do_ds = False
        print("[DEBUG] network.do_ds set to False")
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)
        print(f"[DEBUG] super().validate returned {ret}")
        self.network.do_ds = ds
        print(f"[DEBUG] network.do_ds restored to {ds}")
        return ret


    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        print(f"[DEBUG] predict_preprocessed_data_return_seg_and_softmax called with data=<array>, do_mirroring={do_mirroring}, mirror_axes={mirror_axes}, use_sliding_window={use_sliding_window}, step_size={step_size}, use_gaussian={use_gaussian}, pad_border_mode={pad_border_mode}, pad_kwargs={pad_kwargs}, all_in_gpu={all_in_gpu}, verbose={verbose}, mixed_precision={mixed_precision}")
        ds = self.network.do_ds
        print(f"[DEBUG] saved original do_ds: {ds}")
        self.network.do_ds = False
        print("[DEBUG] network.do_ds set to False")
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """        
        print(f"[DEBUG] super().predict_preprocessed_data_return_seg_and_softmax returned: {ret}")
        self.network.do_ds = ds
        print(f"[DEBUG] network.do_ds restored to {ds}")
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        # print(f"[DEBUG] run_iteration called with data_generator={data_generator}, do_backprop={do_backprop}, run_online_evaluation={run_online_evaluation}")
        data_dict = next(data_generator)
        # print(f"[DEBUG] data_dict obtained: keys={list(data_dict.keys())}")
        data = data_dict['data']
        # print(f"[DEBUG] data extracted: {getattr(data, 'shape', 'unknown')}")
        target = data_dict['target']
        # if isinstance(target, list):
            # print("Here target is list")
            # for idx, item in enumerate(target):
                # print(f"Shape of item {idx} in target is : {getattr(item,'shape','unknown')}")
        # else:
            # print(f"[DEBUG] target is single tensor whose shape is: {getattr(target, 'shape', 'unknown')}")        

        data = maybe_to_torch(data)
        # print(f"[DEBUG] data converted to torch")
        target = maybe_to_torch(target)
        # collapse ISUP >=2 into "lesion"(=1), keep 0 as background
        if isinstance(target, list):
            # target is a list of down-sampled masks at multiple scales
            target = [(t >=2).long() for t in target]
        else:
            target = (target >=2).long()    
        # print(f"[DEBUG] target converted to torch")

        if torch.cuda.is_available():
            # print("[DEBUG] CUDA is available, moving data and target to GPU")
            data = to_cuda(data)
            # print(f"[DEBUG] data on GPU")
            target = to_cuda(target)
            # print(f"[DEBUG] target on GPU")
        # else:
            # print("[DEBUG] CUDA not available, skipping to_cuda")

        # Inspect target values
        # try:
            # t_min, t_max = target.min().item(), target.max().item()
            # print(f"[DEBUG] target min: {t_min}, max: {t_max}, unique: {torch.unique(target)[:10]}")
        # except Exception as e:
            # print(f"[DEBUG] Could not inspect target values: {e}")

        self.optimizer.zero_grad()
        # print("[DEBUG] optimizer.zero_grad() called")

        if self.fp16:
            # print("[DEBUG] fp16 is True, using autocast context")
            with autocast(device_type="cuda"):
                output = self.network(data)
                # if isinstance(output, list):
                    # print("Here output is list")
                    # for idx, item in enumerate(output):
                        # print(f"Shape of item {idx} in output is : {getattr(item,'shape','unknown')}")
                # else:
                    # print(f"[Debug] output is single tensor whose shape is : {getattr(output, 'shape', 'unknown')}")        
                
                del data
                # print("[DEBUG] data deleted")
                l = self.loss(output, target)
                # print(f"[DEBUG] loss computed: {l.item()}")
                # raise SystemExit

            if do_backprop:
                # print("[DEBUG] do_backprop is True, scaling gradient and backprop")
                self.amp_grad_scaler.scale(l).backward()
                # print("[DEBUG] backward called")
                self.amp_grad_scaler.unscale_(self.optimizer)
                # print("[DEBUG] unscale_ called")
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                # print("[DEBUG] gradient clipped")
                self.amp_grad_scaler.step(self.optimizer)
                # print("[DEBUG] amp_grad_scaler.step called")
                self.amp_grad_scaler.update()
                # print("[DEBUG] amp_grad_scaler.update called")
        else:
            # print("[DEBUG] fp16 is False, standard precision training")
            output = self.network(data)
            # print(f"[DEBUG] output computed, shape: {getattr(output, 'shape', 'unknown')}")
            del data
            # print("[DEBUG] data deleted")
            l = self.loss(output, target)
            # print(f"[DEBUG] loss computed: {l}")

            if do_backprop:
                # print("[DEBUG] do_backprop is True, backward() and optimizer.step()")
                l.backward()
                # print("[DEBUG] backward called")
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                # print("[DEBUG] gradient clipped")
                self.optimizer.step()
                # print("[DEBUG] optimizer.step() called")

        if run_online_evaluation:
            # print("[DEBUG] run_online_evaluation is True, calling it")
            self.run_online_evaluation(output, target)
            # print("[DEBUG] run_online_evaluation completed")
        else:
            # print("[DEBUG] run_online_evaluation is False, skipping")
            pass

        del target
        # print("[DEBUG] target deleted")
        result = l.detach().cpu().numpy()
        # print(f"[DEBUG] returning loss numpy: {result}")
        return result


    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """        
        # print(f"[DEBUG] do_split of nnUNetTrainerV2 is called with fold={self.fold}")
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            print("[DEBUG] fold == 'all', using all dataset keys for train and val")
            tr_keys = val_keys = list(self.dataset.keys())
            # print(f"[DEBUG] tr_keys and val_keys set to all keys, count={len(tr_keys)}")
        else:
            # print("[DEBUG] fold != 'all', building split from splits_final.pkl")
            splits_file = join(self.dataset_directory, "splits_final.pkl")
            # print(f"[DEBUG] splits_file path: {splits_file}")
            
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                # print("[DEBUG] splits_file not found, creating new splits")
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                # print(f"[DEBUG] sorted dataset keys: {all_keys_sorted}")
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    # print(f"[DEBUG] fold {i}: computing train and test indices")
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                    # print(f"[DEBUG] splits[{i}] train size={len(train_keys)}, val size={len(test_keys)}")
                save_pickle(splits, splits_file)
                # print("[DEBUG] saved splits to file")
            else:
                # print("[DEBUG] splits_file exists, loading splits")
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                # print(f"[DEBUG] loaded {len(splits)} splits")
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            # print(f"[DEBUG] desired fold: {self.fold}")
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                # print(f"[DEBUG] using predefined split: train size={len(tr_keys)}, val size={len(val_keys)}")
                self.print_to_log_file("This split has %d training and %d validation cases." % (len(tr_keys), len(val_keys)))
            else:
                # print("[DEBUG] desired fold out of range, creating random 80:20 split")
                self.print_to_log_file("INFO: You requested fold %d for training but splits contain only %d folds. I am now creating a random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                # print(f"[DEBUG] random split: train size={len(tr_keys)}, val size={len(val_keys)}")
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases." % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        # print(f"[DEBUG] sorted tr_keys and val_keys")
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        print(f"[DEBUG] dataset_tr populated with {len(self.dataset_tr)} cases")
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]
        print(f"[DEBUG] dataset_val populated with {len(self.dataset_val)} cases")

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """        
        print("[DEBUG] setup_DA_params of nnUNetTrainerV2 is called")
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]
        print(f"[DEBUG] deep_supervision_scales set to {self.deep_supervision_scales}")

        if self.threeD:
            print("[DEBUG] threeD is True, using 3D augmentation params")
            self.data_aug_params = default_3D_augmentation_params
            print("[DEBUG] data_aug_params set to default_3D_augmentation_params")
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            # print(f"[DEBUG] rotation_x set to {self.data_aug_params['rotation_x']}")
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            # print(f"[DEBUG] rotation_y set to {self.data_aug_params['rotation_y']}")
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            # print(f"[DEBUG] rotation_z set to {self.data_aug_params['rotation_z']}")
            if self.do_dummy_2D_aug:
                print("[DEBUG] do_dummy_2D_aug is True, applying 2D augmentation settings")
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = default_2D_augmentation_params["elastic_deform_alpha"]
                # print(f"[DEBUG] elastic_deform_alpha set to {self.data_aug_params['elastic_deform_alpha']}")
                self.data_aug_params["elastic_deform_sigma"] = default_2D_augmentation_params["elastic_deform_sigma"]
                # print(f"[DEBUG] elastic_deform_sigma set to {self.data_aug_params['elastic_deform_sigma']}")
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
                # print(f"[DEBUG] rotation_x overwritten to {self.data_aug_params['rotation_x']}")
        else:
            print("[DEBUG] threeD is False, using 2D augmentation params")
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
                print(f"[DEBUG] rotation_x adjusted for 2D: {default_2D_augmentation_params['rotation_x']}")
            self.data_aug_params = default_2D_augmentation_params
            print("[DEBUG] data_aug_params set to default_2D_augmentation_params")
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm
        print(f"[DEBUG] mask_was_used_for_normalization set to {self.use_mask_for_norm}")

        if self.do_dummy_2D_aug:
            print("[DEBUG] calculating basic_generator_patch_size with dummy 2D")
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            print(f"[DEBUG] basic_generator_patch_size set to {self.basic_generator_patch_size}")
        else:
            print("[DEBUG] calculating basic_generator_patch_size without dummy 2D")
            self.basic_generator_patch_size = get_patch_size(self.patch_size,
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            print(f"[DEBUG] basic_generator_patch_size set to {self.basic_generator_patch_size}")

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        print(f"[DEBUG] scale_range set to {self.data_aug_params['scale_range']}")
        self.data_aug_params["do_elastic"] = False
        print(f"[DEBUG] do_elastic set to {self.data_aug_params['do_elastic']}")
        self.data_aug_params['selected_seg_channels'] = [0]
        print(f"[DEBUG] selected_seg_channels set to {self.data_aug_params['selected_seg_channels']}")
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size
        print(f"[DEBUG] patch_size_for_spatialtransform set to {self.data_aug_params['patch_size_for_spatialtransform']}")
        self.data_aug_params["num_cached_per_thread"] = 2
        print(f"[DEBUG] num_cached_per_thread set to {self.data_aug_params['num_cached_per_thread']}")


    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """        
        print(f"[DEBUG] maybe_update_lr of nnUNetTrainerV2 is called with epoch={epoch}")
        if epoch is None:
            ep = self.epoch + 1
            print(f"[DEBUG] epoch is None, using ep = self.epoch + 1 = {ep}")
        else:
            ep = epoch
            print(f"[DEBUG] using provided epoch ep = {ep}")
        new_lr = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        # print(f"[DEBUG] computed new_lr = {new_lr}")
        self.optimizer.param_groups[0]['lr'] = new_lr
        # print(f"[DEBUG] optimizer.param_groups[0]['lr'] set to {self.optimizer.param_groups[0]['lr']}")
        rounded_lr = np.round(self.optimizer.param_groups[0]['lr'], decimals=6)
        self.print_to_log_file("lr:", rounded_lr)
        print(f"[DEBUG] printed lr to log: {rounded_lr}")

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """        
        # print(f"[DEBUG] on_epoch_end of nnUNetTrainerV2 called at epoch {self.epoch}")
        super().on_epoch_end()
        # print("[DEBUG] super().on_epoch_end() completed")
        continue_training = self.epoch < self.max_num_epochs
        # print(f"[DEBUG] continue_training = {continue_training} (epoch < max_num_epochs)")

        if self.epoch == 100:
            print("[DEBUG] epoch == 100, checking validation metrics")
            if self.all_val_eval_metrics[-1] == 0:
                print("[DEBUG] last validation metric is 0, reducing momentum and reinitializing weights")
                self.optimizer.param_groups[0]["momentum"] = 0.95
                print(f"[DEBUG] optimizer momentum set to {self.optimizer.param_groups[0]['momentum']}")
                self.network.apply(InitWeights_He(1e-2))
                print("[DEBUG] network weights reinitialized with InitWeights_He(1e-2)")
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. Momentum reduced to 0.95 and network weights reinitialized")
        
        # print(f"[DEBUG] on_epoch_end returning continue_training = {continue_training}")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """        
        print(f"[DEBUG] run_training of nnUNetTrainerV2 is called at epoch {self.epoch}")
        self.maybe_update_lr(self.epoch)
        print("[DEBUG] maybe_update_lr executed")
        ds = self.network.do_ds
        print(f"[DEBUG] saved original network.do_ds = {ds}")
        self.network.do_ds = True
        print("[DEBUG] network.do_ds set to True for training")
        ret = super().run_training()
        print(f"[DEBUG] super().run_training returned {ret}")
        self.network.do_ds = ds
        print(f"[DEBUG] network.do_ds restored to {self.network.do_ds}")
        print(f"[DEBUG] run_training returning {ret}")
        return ret
