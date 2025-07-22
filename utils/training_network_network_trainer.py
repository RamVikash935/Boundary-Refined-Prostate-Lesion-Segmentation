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

from _warnings import warn
from typing import Tuple

import matplotlib
from batchgenerators.utilities_file_and_folder_operations import *
from network_architectures_neural_network import SegmentationNetwork
from sklearn.model_selection import KFold
from torch import nn
# from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler
from torch.amp import autocast

from torch.optim.lr_scheduler import _LRScheduler

matplotlib.use("agg")
from time import time, sleep
import torch
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from abc import abstractmethod
from datetime import datetime
from tqdm import trange
from utilities import maybe_to_torch, to_cuda


class NetworkTrainer(object):
    def __init__(self, deterministic=True, fp16=False):
        """
        A generic class that can train almost any neural network (RNNs excluded). It provides basic functionality such
        as the training loop, tracking of training and validation losses (and the target metric if you implement it)
        Training can be terminated early if the validation loss (or the target metric if implemented) do not improve
        anymore. This is based on a moving average (MA) of the loss/metric instead of the raw values to get more smooth
        results.

        What you need to override:
        - __init__
        - initialize
        - run_online_evaluation (optional)
        - finish_online_evaluation (optional)
        - validate
        - predict_test_case
        """        
        self.fp16 = fp16
        # print("Debug: self.fp16 set to", self.fp16)
        self.amp_grad_scaler = None
        print("Debug: self.amp_grad_scaler set to", self.amp_grad_scaler)

        if deterministic:
            np.random.seed(12345)
            print("Debug: numpy random seed set to 12345")
            torch.manual_seed(12345)
            print("Debug: torch manual seed set to 12345")
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(12345)
                print("Debug: torch.cuda.manual_seed_all set to 12345")
            cudnn.deterministic = True
            print("Debug: cudnn.deterministic set to", cudnn.deterministic)
            torch.backends.cudnn.benchmark = False
            print("Debug: torch.backends.cudnn.benchmark set to", torch.backends.cudnn.benchmark)
        else:
            cudnn.deterministic = False
            # print("Debug: cudnn.deterministic set to", cudnn.deterministic)
            torch.backends.cudnn.benchmark = True
            # print("Debug: torch.backends.cudnn.benchmark set to", torch.backends.cudnn.benchmark)

        ################# SET THESE IN self.initialize() ###################################
        self.network: Tuple[SegmentationNetwork, nn.DataParallel] = None
        print("Debug: self.network initialized to", self.network)
        self.optimizer = None
        print("Debug: self.optimizer initialized to", self.optimizer)
        self.lr_scheduler = None
        print("Debug: self.lr_scheduler initialized to", self.lr_scheduler)
        self.tr_gen = self.val_gen = None
        print("Debug: self.tr_gen and self.val_gen initialized to", self.tr_gen, self.val_gen)
        self.was_initialized = False
        print("Debug: self.was_initialized set to", self.was_initialized)

        ################# SET THESE IN INIT ################################################
        self.output_folder = None
        print("Debug: self.output_folder initialized to", self.output_folder)
        self.fold = None
        print("Debug: self.fold initialized to", self.fold)
        self.loss = None
        print("Debug: self.loss initialized to", self.loss)
        self.dataset_directory = None
        print("Debug: self.dataset_directory initialized to", self.dataset_directory)

        ################# SET THESE IN LOAD_DATASET OR DO_SPLIT ############################
        self.dataset = None  # these can be None for inference mode
        print("Debug: self.dataset initialized to", self.dataset)
        self.dataset_tr = self.dataset_val = None
        print("Debug: self.dataset_tr and self.dataset_val initialized to", self.dataset_tr, self.dataset_val)

        ################# THESE DO NOT NECESSARILY NEED TO BE MODIFIED #####################
        self.patience = 50
        print("Debug: self.patience set to", self.patience)
        self.val_eval_criterion_alpha = 0.9
        print("Debug: self.val_eval_criterion_alpha set to", self.val_eval_criterion_alpha)
        self.train_loss_MA_alpha = 0.93
        print("Debug: self.train_loss_MA_alpha set to", self.train_loss_MA_alpha)
        self.train_loss_MA_eps = 5e-4
        print("Debug: self.train_loss_MA_eps set to", self.train_loss_MA_eps)
        self.max_num_epochs = 1000
        print("Debug: self.max_num_epochs set to", self.max_num_epochs)
        self.num_batches_per_epoch = 250
        print("Debug: self.num_batches_per_epoch set to", self.num_batches_per_epoch)
        self.num_val_batches_per_epoch = 50
        print("Debug: self.num_val_batches_per_epoch set to", self.num_val_batches_per_epoch)
        self.also_val_in_tr_mode = False
        print("Debug: self.also_val_in_tr_mode set to", self.also_val_in_tr_mode)
        self.lr_threshold = 1e-6
        print("Debug: self.lr_threshold set to", self.lr_threshold)

        ################# LEAVE THESE ALONE ################################################
        self.val_eval_criterion_MA = None
        print("Debug: self.val_eval_criterion_MA initialized to", self.val_eval_criterion_MA)
        self.train_loss_MA = None
        print("Debug: self.train_loss_MA initialized to", self.train_loss_MA)
        self.best_val_eval_criterion_MA = None
        print("Debug: self.best_val_eval_criterion_MA initialized to", self.best_val_eval_criterion_MA)
        self.best_MA_tr_loss_for_patience = None
        print("Debug: self.best_MA_tr_loss_for_patience initialized to", self.best_MA_tr_loss_for_patience)
        self.best_epoch_based_on_MA_tr_loss = None
        print("Debug: self.best_epoch_based_on_MA_tr_loss initialized to", self.best_epoch_based_on_MA_tr_loss)
        self.all_tr_losses = []
        print("Debug: self.all_tr_losses initialized to", self.all_tr_losses)
        self.all_val_losses = []
        print("Debug: self.all_val_losses initialized to", self.all_val_losses)
        self.all_val_losses_tr_mode = []
        print("Debug: self.all_val_losses_tr_mode initialized to", self.all_val_losses_tr_mode)
        self.all_val_eval_metrics = []
        print("Debug: self.all_val_eval_metrics initialized to", self.all_val_eval_metrics)
        self.epoch = 0
        print("Debug: self.epoch initialized to", self.epoch)
        self.log_file = None
        print("Debug: self.log_file initialized to", self.log_file)
        self.deterministic = deterministic
        print("Debug: self.deterministic set to", self.deterministic)

        self.use_progress_bar = False
        print("Debug: self.use_progress_bar initialized to", self.use_progress_bar)
        if 'nnunet_use_progress_bar' in os.environ.keys():
            self.use_progress_bar = bool(int(os.environ['nnunet_use_progress_bar']))
            print("Debug: self.use_progress_bar updated from env to", self.use_progress_bar)

        ################# Settings for saving checkpoints ##################################
        self.save_every = 50
        print("Debug: self.save_every set to", self.save_every)
        self.save_latest_only = True
        print("Debug: self.save_latest_only set to", self.save_latest_only)
        self.save_intermediate_checkpoints = True
        print("Debug: self.save_intermediate_checkpoints set to", self.save_intermediate_checkpoints)
        self.save_best_checkpoint = True
        print("Debug: self.save_best_checkpoint set to", self.save_best_checkpoint)
        self.save_final_checkpoint = True
        print("Debug: self.save_final_checkpoint set to", self.save_final_checkpoint)


    @abstractmethod
    def initialize(self, training=True):
        """
        create self.output_folder

        modify self.output_folder if you are doing cross-validation (one folder per fold)

        set self.tr_gen and self.val_gen

        call self.initialize_network and self.initialize_optimizer_and_scheduler (important!)

        finally set self.was_initialized to True
        :param training:
        :return:
        """    
        print("Debug: initialize called with training=", training)

    @abstractmethod
    def load_dataset(self):
        print("Debug: load_dataset called")
        pass

    def do_split(self):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)
        :return:
        """        
        splits_file = join(self.dataset_directory, "splits_final.pkl")
        # print("Debug: splits_file path:", splits_file)
        if not isfile(splits_file):
            self.print_to_log_file("Creating new split...")
            # print("Debug: Creating new split, splits_file not found")
            splits = []
            # print("Debug: splits initialized as empty list")
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            # print("Debug: all_keys_sorted:", all_keys_sorted)
            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            # print("Debug: KFold created with n_splits=5, shuffle=True, random_state=12345")
            for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                # print(f"Debug: fold {i}, train_idx={train_idx}, test_idx={test_idx}")
                train_keys = np.array(all_keys_sorted)[train_idx]
                # print("Debug: train_keys:", train_keys)
                test_keys = np.array(all_keys_sorted)[test_idx]
                # print("Debug: test_keys:", test_keys)
                splits.append(OrderedDict())
                print("Debug: appended new OrderedDict to splits")
                splits[-1]['train'] = train_keys
                # print("Debug: set splits[-1]['train']")
                splits[-1]['val'] = test_keys
                # print("Debug: set splits[-1]['val']")
            save_pickle(splits, splits_file)
            print("Debug: saved splits to", splits_file)
        splits = load_pickle(splits_file)
        # print("Debug: loaded splits:", splits)
        
        if self.fold == "all":
            tr_keys = val_keys = list(self.dataset.keys())
            print("Debug: fold='all', tr_keys and val_keys set to all dataset keys")
        else:
            tr_keys = splits[self.fold]['train']
            # print("Debug: tr_keys from splits:", tr_keys)
            val_keys = splits[self.fold]['val']
            # print("Debug: val_keys from splits:", val_keys)
        tr_keys.sort()
        # print("Debug: sorted tr_keys:", tr_keys)
        val_keys.sort()
        # print("Debug: sorted val_keys:", val_keys)
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
            # print(f"Debug: dataset_tr[{i}] set")
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]
            # print(f"Debug: dataset_val[{i}] set")

    def plot_progress(self):
        """
        Should probably by improved
        :return:
        """    
        try:
            font = {'weight': 'normal', 'size': 18}
            # print("Debug: font configured:", font)
            matplotlib.rc('font', **font)
            # print("Debug: matplotlib rc font set")
            fig = plt.figure(figsize=(30, 24))
            # print("Debug: figure created with figsize=(30,24)")
            ax = fig.add_subplot(111)
            # print("Debug: main axis created")
            ax2 = ax.twinx()
            # print("Debug: secondary axis created")

            x_values = list(range(self.epoch + 1))
            # print("Debug: x_values:", x_values)

            ax.plot(x_values, self.all_tr_losses, ls='-', label="loss_tr")
            # print("Debug: plotted all_tr_losses")

            ax.plot(x_values, self.all_val_losses, ls='-', label="loss_val, train=False")
            # print("Debug: plotted all_val_losses")

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, ls='-', label="loss_val, train=True")
                # print("Debug: plotted all_val_losses_tr_mode")
            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics, ls='--', label="evaluation metric")
                # print("Debug: plotted all_val_eval_metrics")

            ax.set_xlabel("epoch")
            # print("Debug: set x label")
            ax.set_ylabel("loss")
            # print("Debug: set loss y label")
            ax2.set_ylabel("evaluation metric")
            # print("Debug: set evaluation metric y label")
            ax.legend()
            # print("Debug: legend on ax")
            ax2.legend(loc=9)
            # print("Debug: legend on ax2")

            fig.savefig(join(self.output_folder, "progress.png"))
            # print("Debug: saved figure to progress.png")
            plt.close()
            # print("Debug: closed plt")
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())
            # print("Debug: IOError in plot_progress, logged error")


    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        timestamp = time()
        # print("Debug: timestamp generated:", timestamp)
        dt_object = datetime.fromtimestamp(timestamp)
        # print("Debug: dt_object from timestamp:", dt_object)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)
            # print("Debug: args updated with timestamp:", args)

        if self.log_file is None:
            maybe_mkdir_p(self.output_folder)
            # print("Debug: ensured output_folder exists:", self.output_folder)
            timestamp = datetime.now()
            # print("Debug: timestamp reset to now:", timestamp)
            self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
            print("Debug: self.log_file set to", self.log_file)
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
                # print("Debug: wrote initial 'Starting...' to log file")
        successful = False
        # print("Debug: successful flag initialized to", successful)
        max_attempts = 5
        # print("Debug: max_attempts set to", max_attempts)
        ctr = 0
        # print("Debug: ctr initialized to", ctr)
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        # print("Debug: logged arg to file:", a)
                        f.write(" ")
                        # print("Debug: logged space to file")
                    f.write("\n")
                    # print("Debug: logged newline to file")
                successful = True
                # print("Debug: successful set to", successful)
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                print("Debug: IOError caught while logging, retrying")
                sleep(0.5)
                print("Debug: sleep for retry")
                ctr += 1
                print("Debug: ctr incremented to", ctr)
                
        if also_print_to_console:
            print(*args)
            # print("Debug: printed args to console")

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        # print("Debug: save_checkpoint start_time:", start_time)
        state_dict = self.network.state_dict()
        # print("Debug: fetched network state_dict")
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
            # print(f"Debug: moved state_dict['{key}'] to CPU")
        lr_sched_state_dct = None
        # print("Debug: lr_sched_state_dct initialized to None")
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'state_dict'):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            # print("Debug: fetched lr_scheduler state_dict")
            # WTF is this!?
            # for key in lr_sched_state_dct.keys():
            #    lr_sched_state_dct[key] = lr_sched_state_dct[key]            
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
            # print("Debug: fetched optimizer state_dict")
        else:
            optimizer_state_dict = None
            # print("Debug: optimizer_state_dict set to None")

        self.print_to_log_file("saving checkpoint...")
        print("Debug: called print_to_log_file for saving checkpoint")
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics),
            'best_stuff': (self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA)
        }
        print("Debug: prepared save_this dict with keys:", list(save_this.keys()))
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()
            # print("Debug: added amp_grad_scaler to save_this")

        torch.save(save_this, fname)
        print("Debug: torch.save called with fname:", fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))
        # print("Debug: print_to_log_file called for completion message")

    def load_best_checkpoint(self, train=True):
        print("Debug: load_best_checkpoint called with train=", train)
        if self.fold is None:
            print("Debug: self.fold is None, raising RuntimeError")
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        print("Debug: self.fold is", self.fold)
        best_path = join(self.output_folder, "model_best.model")
        print("Debug: best_path set to", best_path)
        if isfile(best_path):
            print("Debug: best checkpoint exists, loading")
            self.load_checkpoint(best_path, train=train)
            print("Debug: load_checkpoint called for best checkpoint")
        else:
            self.print_to_log_file("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling back to load_latest_checkpoint")
            print("Debug: warning logged for missing best checkpoint")
            self.load_latest_checkpoint(train)
            print("Debug: load_latest_checkpoint called as fallback")

    def load_latest_checkpoint(self, train=True):
        print("Debug: load_latest_checkpoint called with train=", train)
        final_path = join(self.output_folder, "model_final_checkpoint.model")
        print("Debug: checking for final checkpoint at", final_path)
        if isfile(final_path):
            print("Debug: found final checkpoint, loading")
            return self.load_checkpoint(final_path, train=train)
        latest_path = join(self.output_folder, "model_latest.model")
        print("Debug: checking for latest checkpoint at", latest_path)
        if isfile(latest_path):
            print("Debug: found latest checkpoint, loading")
            return self.load_checkpoint(latest_path, train=train)
        best_path = join(self.output_folder, "model_best.model")
        print("Debug: checking for best checkpoint at", best_path)
        if isfile(best_path):
            print("Debug: found best checkpoint, loading best")
            return self.load_best_checkpoint(train)
        print("Debug: no checkpoint found, raising RuntimeError")
        raise RuntimeError("No checkpoint found")

    def load_final_checkpoint(self, train=False):
        print("Debug: load_final_checkpoint called with train=", train)
        filename = join(self.output_folder, "model_final_checkpoint.model")
        print("Debug: final checkpoint filename=", filename)
        if not isfile(filename):
            print("Debug: final checkpoint not found, raising RuntimeError")
            raise RuntimeError("Final checkpoint not found. Expected: %s. Please finish the training first." % filename)
        print("Debug: final checkpoint exists, loading")
        return self.load_checkpoint(filename, train=train)

    def load_checkpoint(self, fname, train=True):
        print("Debug: load_checkpoint called with fname=", fname, "train=", train)
        self.print_to_log_file("loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            print("Debug: was_initialized=False, calling initialize")
            self.initialize(train)
        print("Debug: loading saved_model from", fname)
        saved_model = torch.load(fname, map_location=torch.device('cpu'), weights_only=False)
        print("Debug: loaded saved_model")
        self.load_checkpoint_ram(saved_model, train)
        print("Debug: load_checkpoint_ram completed")

    @abstractmethod
    def initialize_network(self):
        """
        initialize self.network here
        :return:
        """        
        print("Debug: initialize_network called")
        pass

    @abstractmethod
    def initialize_optimizer_and_scheduler(self):
        """
        initialize self.optimizer and self.lr_scheduler (if applicable) here
        :return:
        """        
        print("Debug: initialize_optimizer_and_scheduler called")
        pass

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """        
        print("Debug: load_checkpoint_ram called with train=", train)
        if not self.was_initialized:
            print("Debug: was_initialized=False inside load_checkpoint_ram, calling initialize")
            self.initialize(train)

        new_state_dict = OrderedDict()
        print("Debug: initialized new_state_dict")
        curr_keys = list(self.network.state_dict().keys())
        print("Debug: current network state_dict keys:", curr_keys)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match        
        for k, value in checkpoint['state_dict'].items():
            print(f"Debug: processing checkpoint key {k}")
            key = k
            if key not in curr_keys and key.startswith('module.'):
                key = key[7:]
                print(f"Debug: stripped 'module.' prefix, new key={key}")
            new_state_dict[key] = value
            print(f"Debug: set new_state_dict[{key}]")

        if self.fp16:
            print("Debug: fp16 enabled, initializing amp if needed")
            self._maybe_init_amp()
            if train and 'amp_grad_scaler' in checkpoint:
                self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])
                print("Debug: loaded amp_grad_scaler state_dict")

        print("Debug: loading state_dict into network")
        self.network.load_state_dict(new_state_dict)
        print("Debug: network.load_state_dict completed")
        self.epoch = checkpoint['epoch']
        print("Debug: epoch set to", self.epoch)
        if train:
            opt_state = checkpoint.get('optimizer_state_dict')
            print("Debug: optimizer_state_dict loaded:", opt_state)
            if opt_state is not None:
                self.optimizer.load_state_dict(opt_state)
                print("Debug: optimizer.load_state_dict completed")

            if self.lr_scheduler and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint.get('lr_scheduler_state_dict'):
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                print("Debug: lr_scheduler.load_state_dict completed")

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)
                print("Debug: lr_scheduler.step called with epoch", self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint['plot_stuff']
        print("Debug: restored plot_stuff arrays")

        # load best loss (if present)
        if 'best_stuff' in checkpoint:
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = checkpoint['best_stuff']
            print("Debug: restored best_stuff values")

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses)...")
            print("Debug: epoch mismatch, correcting")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]
            print("Debug: truncated loss and metric arrays to epoch length")

        self._maybe_init_amp()
        print("Debug: final _maybe_init_amp call completed")


    def _maybe_init_amp(self):
        # print("Debug: _maybe_init_amp called")
        if self.fp16 and self.amp_grad_scaler is None:
            self.amp_grad_scaler = GradScaler()
            # print("Debug: amp_grad_scaler initialized to GradScaler instance")

    def plot_network_architecture(self):
        """
        can be implemented (see nnUNetTrainer) but does not have to. Not implemented here because it imposes stronger
        assumptions on the presence of class variables
        :return:
        """        
        print("Debug: plot_network_architecture called")
        pass

    def run_training(self):
        print("Debug: run_training called")
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")
            print("Debug: logged CPU training warning to log file")
            
        _ = self.tr_gen.next()
        print("Debug: fetched next batch from tr_gen")

        _ = self.val_gen.next()
        print("Debug: fetched next batch from val_gen")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Debug: emptied CUDA cache")

        self._maybe_init_amp()
        # print("Debug: called _maybe_init_amp")

        maybe_mkdir_p(self.output_folder)
        print("Debug: ensured output_folder exists:", self.output_folder)

        self.plot_network_architecture()
        print("Debug: called plot_network_architecture")

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! If you want deterministic then set benchmark=False")
            print("Debug: warned about cudnn benchmark/deterministic conflict")

        if not self.was_initialized:
            self.initialize(True)
            print("Debug: initialize(True) called because was_initialized=False")

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            print("Debug: logging epoch start", self.epoch)
            epoch_start_time = time()
            # print("Debug: epoch_start_time set to", epoch_start_time)
            train_losses_epoch = []
            # print("Debug: initialized train_losses_epoch list")

            # train one epoch
            self.network.train()
            print("Debug: set network to train mode")

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description(f"Epoch {self.epoch+1}/{self.max_num_epochs}")
                        print(f"Debug: progress bar description set for batch {b}")
                        l = self.run_iteration(self.tr_gen, True)
                        # print("Debug: run_iteration returned", l)
                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
                        # print("Debug: appended loss to train_losses_epoch")
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration(self.tr_gen, True)
                    # print("Debug: run_iteration returned", l)
                    train_losses_epoch.append(l)
                    # print("Debug: appended loss to train_losses_epoch")
                    # raise SystemExit
                
            self.all_tr_losses.append(np.mean(train_losses_epoch))
            print("Debug: appended mean train loss", self.all_tr_losses[-1])
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])
            # print("Debug: logged train loss")

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                # print("Debug: set network to eval mode")
                val_losses = []
                # print("Debug: initialized val_losses list")
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    # print("Debug: run_iteration for val_gen returned", l)
                    val_losses.append(l)
                    # print("Debug: appended loss to val_losses")
                    
                self.all_val_losses.append(np.mean(val_losses))
                print("Debug: appended mean validation loss", self.all_val_losses[-1])
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])
                # print("Debug: logged validation loss")

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    print("Debug: set network to train mode for also_val_in_tr_mode")
                    val_losses_tr = []
                    print("Debug: initialized val_losses_tr list")
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        print("Debug: run_iteration for val_gen (train mode) returned", l)
                        val_losses_tr.append(l)
                        print("Debug: appended loss to val_losses_tr")
                    self.all_val_losses_tr_mode.append(np.mean(val_losses_tr))
                    print("Debug: appended mean validation loss (train mode)", self.all_val_losses_tr_mode[-1])
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])
                    print("Debug: logged validation loss (train mode)")

            self.update_train_loss_MA()
            # print("Debug: update_train_loss_MA called")
            continue_training = self.on_epoch_end()
            print("Debug: on_epoch_end returned", continue_training)

            epoch_end_time = time()
            # print("Debug: epoch_end_time set to", epoch_end_time)

            if not continue_training:
                print("Debug: stopping training loop early")
                break

            self.epoch += 1
            print("Debug: incremented epoch to", self.epoch)
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))
            # print("Debug: logged epoch duration")

        self.epoch -= 1
        print("Debug: decremented epoch to", self.epoch)

        if self.save_final_checkpoint:
            print("Debug: save_final_checkpoint is True, saving final checkpoint")
            self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
            print("Debug: final checkpoint saved")
            # now we can delete latest as it will be identical with final
            
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
            print("Debug: removed model_latest.model")
            
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))
            print("Debug: removed model_latest.model.pkl")



    def maybe_update_lr(self):
        # print("Debug: maybe_update_lr called")
        # maybe update learning rate
        if self.lr_scheduler is not None:
            # print("Debug: lr_scheduler is not None")
            assert isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau, lr_scheduler._LRScheduler))
            # print("Debug: lr_scheduler type assertion passed")
            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                # print(f"Debug: ReduceLROnPlateau stepping with train_loss_MA={self.train_loss_MA}")
                self.lr_scheduler.step(self.train_loss_MA)
            else:
                # print(f"Debug: _LRScheduler stepping with epoch+1={self.epoch + 1}")
                self.lr_scheduler.step(self.epoch + 1)
        else:
            print("Debug: lr_scheduler is None")
            
        self.print_to_log_file(f"lr is now (scheduler) {self.optimizer.param_groups[0]['lr']}")
        # print(f"Debug: logged lr {self.optimizer.param_groups[0]['lr']} to log file")

    def maybe_save_checkpoint(self):
        # print("Debug: maybe_save_checkpoint called")
        """
        Saves a checkpoint every save_every epochs.
        :return:
        """
        if self.save_intermediate_checkpoints and (self.epoch % self.save_every == (self.save_every - 1)):
            print(f"Debug: epoch {self.epoch} matches save condition (save_every={self.save_every})")
            self.print_to_log_file("saving scheduled checkpoint file...")
            # print("Debug: logged scheduled checkpoint message")
            if not self.save_latest_only:
                fname = join(self.output_folder, f"model_ep_{self.epoch+1:03d}.model")
                self.save_checkpoint(fname)
                print(f"Debug: saved epoch-specific checkpoint to {fname}")
            
            latest = join(self.output_folder, "model_latest.model")
            self.save_checkpoint(latest)
            print(f"Debug: saved latest checkpoint to {latest}")
            self.print_to_log_file("done")
            # print("Debug: logged 'done' to log file")
        else:
            print(f"Debug: save condition not met (save_intermediate_checkpoints={self.save_intermediate_checkpoints}, epoch={self.epoch})")

    def update_eval_criterion_MA(self):    
        """
        If self.all_val_eval_metrics is unused (len=0) then we fall back to using -self.all_val_losses for the MA to determine early stopping
        (not a minimization, but a maximization of a metric and therefore the - in the latter case)
        :return:
        """        
        # print("Debug: update_eval_criterion_MA called")
        if self.val_eval_criterion_MA is None:
            # print("Debug: val_eval_criterion_MA is None")
            if len(self.all_val_eval_metrics) == 0:
                self.val_eval_criterion_MA = -self.all_val_losses[-1]
                # print(f"Debug: set val_eval_criterion_MA to -last val_loss={self.val_eval_criterion_MA}")
            else:
                self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
                # print(f"Debug: set val_eval_criterion_MA to last eval metric={self.val_eval_criterion_MA}")
        else:
            print(f"Debug: existing val_eval_criterion_MA={self.val_eval_criterion_MA}")
            if len(self.all_val_eval_metrics) == 0:
                old = self.val_eval_criterion_MA
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * old - \
                                             (1 - self.val_eval_criterion_alpha) * self.all_val_losses[-1]
                print(f"Debug: updated val_eval_criterion_MA from {old} to {self.val_eval_criterion_MA} using val loss")
            else:
                old = self.val_eval_criterion_MA
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * old + \
                                             (1 - self.val_eval_criterion_alpha) * self.all_val_eval_metrics[-1]
                print(f"Debug: updated val_eval_criterion_MA from {old} to {self.val_eval_criterion_MA} using eval metric")

    def manage_patience(self):
        # update patience
        # print("Debug: manage_patience called")
        continue_training = True
        # print(f"Debug: initial continue_training={continue_training}")
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them            
            # print(f"Debug: patience={self.patience}")
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                print(f"Debug: initialized best_MA_tr_loss_for_patience={self.best_MA_tr_loss_for_patience}")
            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                print(f"Debug: initialized best_epoch_based_on_MA_tr_loss={self.best_epoch_based_on_MA_tr_loss}")
            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                print(f"Debug: initialized best_val_eval_criterion_MA={self.best_val_eval_criterion_MA}")

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            #self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            #self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)
            
            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                print(f"Debug: new best_val_eval_criterion_MA={self.best_val_eval_criterion_MA}")
                if self.save_best_checkpoint:
                    self.save_checkpoint(join(self.output_folder, "model_best.model"))
                    # print("Debug: saved best checkpoint")

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                print(f"Debug: new best_MA_tr_loss_for_patience={self.best_MA_tr_loss_for_patience} at epoch={self.best_epoch_based_on_MA_tr_loss}")
            else:
                print(f"Debug: no improvement: train_loss_MA={self.train_loss_MA}, best={self.best_MA_tr_loss_for_patience}")

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                print(f"Debug: (epoch - best_epoch)={self.epoch - self.best_epoch_based_on_MA_tr_loss} exceeded patience")
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                    print(f"Debug: lr above threshold, resetting best_epoch to {self.best_epoch_based_on_MA_tr_loss}")
                else:
                    continue_training = False
                    print("Debug: patience exceeded and lr below threshold, stopping training")
            else:
                print(f"Debug: patience status: {self.epoch - self.best_epoch_based_on_MA_tr_loss}/{self.patience}")
        
        print(f"Debug: manage_patience returning {continue_training}")
        return continue_training


    def on_epoch_end(self):
        # print("Debug: on_epoch_end called")
        self.finish_online_evaluation()
        # print("Debug: finish_online_evaluation called")
        self.plot_progress()
        # print("Debug: plot_progress called")
        self.maybe_update_lr()
        # print("Debug: maybe_update_lr called")
        self.maybe_save_checkpoint()
        # print("Debug: maybe_save_checkpoint called")
        self.update_eval_criterion_MA()
        # print("Debug: update_eval_criterion_MA called")
        continue_training = self.manage_patience()
        # print(f"Debug: manage_patience returned {continue_training}")
        return continue_training

    def update_train_loss_MA(self):
        # print("Debug: update_train_loss_MA called")
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
            print(f"Debug: train_loss_MA initialized to {self.train_loss_MA}")
        else:
            old = self.train_loss_MA
            new = self.train_loss_MA_alpha * old + (1 - self.train_loss_MA_alpha) * self.all_tr_losses[-1]
            self.train_loss_MA = new
            print(f"Debug: train_loss_MA updated from {old} to {new} using last train loss {self.all_tr_losses[-1]}")

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        # print("Debug: run_iteration called with do_backprop=", do_backprop, "run_online_evaluation=", run_online_evaluation)
        data_dict = next(data_generator)
        # print("Debug: data_dict fetched")
        data = data_dict['data']
        target = data_dict['target']
        # print("Debug: extracted data and target from data_dict")
        data = maybe_to_torch(data)
        # print("Debug: data converted to torch tensor")
        target = maybe_to_torch(target)
        # print("Debug: target converted to torch tensor")
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            # print("Debug: data and target moved to CUDA")
        self.optimizer.zero_grad()
        # print("Debug: optimizer gradients zeroed")
        if self.fp16:
            with autocast(device_type="cuda"):
                output = self.network(data)
                # print("Debug: network output computed under autocast")
                del data
                l = self.loss(output, target)
                # print("Debug: loss computed")
            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                # print("Debug: backward pass scaled with amp_grad_scaler")
                self.amp_grad_scaler.step(self.optimizer)
                # print("Debug: optimizer step with amp_grad_scaler")
                self.amp_grad_scaler.update()
                # print("Debug: amp_grad_scaler updated")
        else:
            output = self.network(data)
            print("Debug: network output computed")
            del data
            l = self.loss(output, target)
            print("Debug: loss computed")
            
            if do_backprop:
                l.backward()
                print("Debug: backward pass")
                self.optimizer.step()
                print("Debug: optimizer step")
                
        if run_online_evaluation:
            self.run_online_evaluation(output, target)
            print("Debug: run_online_evaluation called")
            
        del target
        # print("Debug: target deleted")
        l_np = l.detach().cpu().numpy()
        # print("Debug: loss detached to numpy:", l_np)
        return l_np

    def run_online_evaluation(self, *args, **kwargs):
        """
        Can be implemented, does not have to
        :param output_torch:
        :param target_npy:
        :return:
        """        
        print("Debug: run_online_evaluation stub called with args=", args, "kwargs=", kwargs)
        pass



    def finish_online_evaluation(self):
        """
        Can be implemented, does not have to
        :return:
        """        
        print("Debug: finish_online_evaluation called")
        pass

    @abstractmethod
    def validate(self, *args, **kwargs):
        print("Debug: validate called with args=", args, "kwargs=", kwargs)
        pass

    def find_lr(self, num_iters=1000, init_value=1e-6, final_value=10., beta=0.98):
        """
        stolen and adapted from here: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
        :param num_iters:
        :param init_value:
        :param final_value:
        :param beta:
        :return:
        """        
        print("Debug: find_lr called with num_iters=", num_iters, "init_value=", init_value, "final_value=", final_value, "beta=", beta)
        import math
        # print("Debug: imported math")
        self._maybe_init_amp()
        print("Debug: called _maybe_init_amp")
        mult = (final_value / init_value) ** (1 / num_iters)
        # print("Debug: computed mult=", mult)
        lr = init_value
        # print("Debug: initial lr=", lr)
        self.optimizer.param_groups[0]['lr'] = lr
        print("Debug: set optimizer lr to", lr)
        avg_loss = 0.
        print("Debug: initialized avg_loss=", avg_loss)
        best_loss = 0.
        print("Debug: initialized best_loss=", best_loss)
        losses = []
        # print("Debug: initialized losses list")
        log_lrs = []
        # print("Debug: initialized log_lrs list")

        for batch_num in range(1, num_iters + 1):
            # +1 because this one here is not designed to have negative loss...            
            print("Debug: find_lr loop batch_num=", batch_num)
            loss = self.run_iteration(self.tr_gen, do_backprop=True, run_online_evaluation=False)
            try:
                raw_loss = loss.data.item()
            except AttributeError:
                raw_loss = loss
            loss = raw_loss + 1
            print(f"Debug: raw loss= {raw_loss}, plus 1 => {loss}")
            
            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss
            print("Debug: updated avg_loss=", avg_loss)
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            print("Debug: computed smoothed_loss=", smoothed_loss)

            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                print("Debug: smoothed_loss > 4*best_loss, breaking")
                break
            
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
                print("Debug: updated best_loss=", best_loss)

            # Store the values
            losses.append(smoothed_loss)
            # print("Debug: appended smoothed_loss to losses")
            log_lrs.append(math.log10(lr))
            print("Debug: appended log10(lr) to log_lrs=", math.log10(lr))

            # Update the lr for the next step
            lr *= mult
            # print("Debug: updated lr to", lr)
            self.optimizer.param_groups[0]['lr'] = lr
            print("Debug: set optimizer lr to", lr)

        import matplotlib.pyplot as plt
        # print("Debug: imported matplotlib.pyplot as plt")
        lrs_vals = [10 ** i for i in log_lrs]
        # print("Debug: computed lrs_vals from log_lrs")
        fig = plt.figure()
        # print("Debug: created figure")
        plt.xscale('log')
        # print("Debug: set xscale to log")
        plt.plot(lrs_vals[10:-5], losses[10:-5])
        # print("Debug: plotted lrs vs losses")
        plt.savefig(join(self.output_folder, "lr_finder.png"))
        # print("Debug: saved lr_finder.png")
        plt.close()
        # print("Debug: closed plt")
        return log_lrs, losses
