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
#    limitations under the License

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

from all_helper_fn_n_class.batchgenerators.utilities_file_and_folder_operations import *
from all_helper_fn_n_class.run_default_configuration import get_default_configuration
from all_helper_fn_n_class.paths import default_plans_identifier
from all_helper_fn_n_class.run_load_pretrained_weights import load_pretrained_weights
from all_helper_fn_n_class.training_cascade_stuff_predict_next_stage import predict_next_stage
from all_helper_fn_n_class.training_network_variants_architectural_nnUNetTrainerV2_LearnableEASAG_AttentionResidualDecoderUNet import nnUNetTrainerV2_Larnable_EASAG_Residual_DecoderUNet
from all_helper_fn_n_class.training_network_nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from all_helper_fn_n_class.training_network_nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from all_helper_fn_n_class.utilities import convert_id_to_task_name

def train_network(
    network: str,
    network_trainer: str,
    task: Union[str, int],
    fold: Union[int, str],
    validation_only: bool = False,
    continue_training: bool = False,
    plans_identifier: str = 'nnUNetPlans_Larnable_EASAG_ResidualDecoder_AttentionUNet_v2.1',
    use_compressed_data: bool = False,
    deterministic: bool = False,
    export_npz: bool = False,
    find_lr: bool = False,
    valbest: bool = False,
    fp32: bool = False,
    validation_folder_name: str = 'validation_raw',
    disable_saving: bool = False,
    disable_postprocessing_on_folds: bool = False,
    val_disable_overwrite: bool = True,
    disable_next_stage_pred: bool = False,
    pretrained_weights: Optional[str] = None  # use Optional for None defaults
):    
    """
    Train or validate an nnU-Net model (simplified interface).

    Parameters:
    - network: '2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'.
    - network_trainer: Name of the nnU-Net Trainer class to use.
    - task: Task name ('TaskXXX_') or integer task ID.
    - fold: Integer fold (0-5) or 'all'.
    - validation_only: If True, only run validation.
    - continue_training: If True, continue from last checkpoint.
    - plans_identifier: Identifier for planning (default).
    - use_compressed_data: If True, read data without decompressing.
    - deterministic: If True, run deterministic training (slower).
    - export_npz: If True, export softmax as .npz during validation.
    - find_lr: If True, run learning rate finder and exit.
    - valbest: If True, validate using the best checkpoint.
    - fp32: If True, disable mixed precision (use fp32).
    - validation_folder_name: Name of the validation folder.
    - disable_saving: If True, do not save any checkpoints.
    - disable_postprocessing_on_folds: If True, skip per-fold postprocessing.
    - val_disable_overwrite: If False, allow overwriting validation outputs.
    - disable_next_stage_pred: If True, do not predict next stage for 3d_lowres.
    - pretrained_weights: Path to pretrained weights (.model) to load.
    """    
    # Normalize task
    if not str(task).startswith('Task'):
        task = convert_id_to_task_name(int(task))
        
    # Normalize fold
    if fold != 'all':
        fold = int(fold)
        
    # Determine precision and data unpacking
    run_mixed_precision = not fp32
    
    decompress_data = not use_compressed_data
    
    # Retrieve default config
    plans_file, output_folder, dataset_dir, batch_dice, stage, trainer_cls = get_default_configuration(network, task, network_trainer, plans_identifier)
    
    if trainer_cls is None:
        raise RuntimeError(f"Trainer class '{network_trainer}' not found.")

    # Validate cascade trainer if needed
    if network == '3d_cascade_fullres':
        assert issubclass(trainer_cls, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes)), "For '3d_cascade_fullres', trainer must derive from nnUNetTrainerCascadeFullRes"
    else:
        # assert issubclass(trainer_cls, nnUNetTrainer), f"Trainer '{network_trainer}' is not a subclass of nnUNetTrainer"

    # Instantiate trainer
    trainer = trainer_cls(
        plans_file,
        fold,
        output_folder=output_folder,
        dataset_directory=dataset_dir,
        batch_dice=batch_dice,
        stage=stage,
        unpack_data=decompress_data,
        deterministic=deterministic,
        fp16=run_mixed_precision
    )
    
    # Disable checkpoint saving if requested
    if disable_saving:
        trainer.save_final_checkpoint = False
        trainer.save_best_checkpoint = False
        trainer.save_intermediate_checkpoints = True
        trainer.save_latest_only = True
        
    # Initialize training or validation
    trainer.initialize(not validation_only)
    
    # Learning rate finder
    if find_lr:
        trainer.find_lr()
        return

    # Training or validation flow
    if not validation_only:
        if continue_training:
            trainer.load_latest_checkpoint()
        elif pretrained_weights is not None:
            load_pretrained_weights(trainer.network, pretrained_weights)
            
        trainer.run_training()
    else:
        if valbest:
            trainer.load_best_checkpoint(train=False)
        else:
            trainer.load_final_checkpoint(train=False)
        
        trainer.network.eval()
        trainer.validate(
            save_softmax=export_npz,
            validation_folder_name=validation_folder_name,
            run_postprocessing_on_folds=not disable_postprocessing_on_folds,
            overwrite=val_disable_overwrite
        )
    
    # Optional next-stage prediction
    if network == '3d_lowres' and not disable_next_stage_pred:
        predict_next_stage(
            trainer,
            join(dataset_dir, trainer.plans['data_identifier'] + '_stage1')
        )
        
train_network('3d_fullres', 'nnUNetTrainerV2_Larnable_EASAG_Residual_DecoderUNet',5,4)        