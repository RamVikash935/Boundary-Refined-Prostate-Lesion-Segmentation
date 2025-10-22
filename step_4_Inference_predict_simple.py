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

from all_helper_fn_n_class.inference_predict import predict_from_folder
from all_helper_fn_n_class.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from all_helper_fn_n_class.batchgenerators.utilities_file_and_folder_operations import join, isdir
from all_helper_fn_n_class.utilities import convert_id_to_task_name


def run_inference(
    input_folder: str,
    output_folder: str,
    task: str| int,
    trainer_class_name: str = default_trainer,
    cascade_trainer_class_name: str = default_cascade_trainer,
    model: str = "3d_fullres",
    plans_identifier: str = default_plans_identifier,
    folds: list[int] | str | None = 'None',
    save_npz: bool = False,
    lowres_segmentations: str | None = 'None',
    part_id: int = 0,
    num_parts: int = 1,
    num_threads_preprocessing: int = 6,
    num_threads_nifti_save: int = 2,
    disable_tta: bool = False,
    overwrite_existing: bool = False,
    mode: str = "normal",
    all_in_gpu: str | bool | None = 'None',
    step_size: float = 0.5,
    checkpoint_name: str = 'model_final_checkpoint',
    disable_mixed_precision: bool = False
):
    """
    Run nnU-Net inference on a folder of images.

    Parameters:
    - input_folder: Path containing all modalities per case, named CASENAME_XXXX.nii.gz.
    - output_folder: Path where predictions are saved.
    - task: Task identifier ("TaskXXX_" string or integer ID).
    - trainer_class_name: Trainer for 2D, fullres-3D, lowres-3D U-Nets.
    - cascade_trainer_class_name: Trainer for cascade full-resolution stage.
    - model: One of ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'].
    - plans_identifier: Plans identifier for model folder naming.
    - folds: List of fold numbers or 'all' or None to detect automatically.
    - save_npz: If True, store softmax as .npz for ensembling.
    - lowres_segmentations: Path to lowres preds for cascade; None to auto-run lowres first.
    - part_id, num_parts: For multi-GPU parallelization.
    - num_threads_preprocessing: Parallel threads for preprocessing.
    - num_threads_nifti_save: Parallel threads for nifti export.
    - disable_tta: Disable test-time augmentation if True.
    - overwrite_existing: Overwrite existing predictions if True.
    - mode: Prediction mode (internal).
    - all_in_gpu: None, False, or True to control GPU usage.
    - step_size: Step size for sliding window.
    - checkpoint_name: Checkpoint to load (default final).
    - disable_mixed_precision: Disable AMP if True.
    """
    # Normalize task name
    if isinstance(task, int) or not str(task).startswith("Task"):
        task = convert_id_to_task_name(int(task))

    # Validate model
    assert model in ["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"], \
        "model must be one of ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres']"

    # Normalize lowres_segmentations
    if lowres_segmentations == 'None':
        lowres_segmentations = None

    # Normalize folds
    if isinstance(folds, list):
        if folds == ['all']:
            folds = None
        else:
            folds = [int(f) for f in folds]
    elif folds == 'None':
        folds = None
    else:
        raise ValueError(f"Unexpected folds value: {folds}")

    # Normalize all_in_gpu
    if isinstance(all_in_gpu, str):
        if all_in_gpu == 'None':
            all_in_gpu = None
        elif all_in_gpu == 'True':
            all_in_gpu = True
        elif all_in_gpu == 'False':
            all_in_gpu = False
        else:
            raise ValueError(f"Unexpected all_in_gpu value: {all_in_gpu}")

    # Handle cascade lowres first
    if model == '3d_cascade_fullres' and lowres_segmentations is None:
        assert part_id == 0 and num_parts == 1, "Must run lowres stage separately if parallel parts are used"
        lowres_model_folder = join(network_training_output_dir, '3d_lowres', task,f"{trainer_class_name}__{plans_identifier}")
        assert isdir(lowres_model_folder), f"Lowres model folder not found: {lowres_model_folder}"
        lowres_output_folder = join(output_folder, '3d_lowres_predictions')
        predict_from_folder(
            lowres_model_folder, input_folder, lowres_output_folder,
            folds, False, num_threads_preprocessing,
            num_threads_nifti_save, None,
            part_id, num_parts,
            not disable_tta,
            overwrite_existing=overwrite_existing,
            mode=mode,
            overwrite_all_in_gpu=all_in_gpu,
            mixed_precision=not disable_mixed_precision,
            step_size=step_size
        )
        lowres_segmentations = lowres_output_folder
        torch.cuda.empty_cache()

    # Choose trainer
    trainer = cascade_trainer_class_name if model == '3d_cascade_fullres' else trainer_class_name

    # Construct model folder
    model_folder = join(network_training_output_dir, model, task,f"{trainer}__{plans_identifier}")
    assert isdir(model_folder), f"Model folder not found: {model_folder}"

    # Run prediction
    return predict_from_folder(
        model_folder, input_folder, output_folder,
        folds, save_npz, num_threads_preprocessing,
        num_threads_nifti_save, lowres_segmentations,
        part_id, num_parts,
        not disable_tta,
        overwrite_existing=overwrite_existing,
        mode=mode,
        overwrite_all_in_gpu=all_in_gpu,
        mixed_precision=not disable_mixed_precision,
        step_size=step_size,
        checkpoint_name=checkpoint_name
    )


input_folder= '/home/ss_students/zywx/raw/nnUNet_raw_data/Task005_Prostate/imagesTs'
output_folder= '/home/ss_students/zywx/inference_result/prediction'
task=5
trainer_class_name = 'nnUNetTrainerV2_Larnable_EASAG_Residual_DecoderUNet'
cascade_trainer_class_name = default_cascade_trainer
model = "3d_fullres"
plans_identifier = 'nnUNetPlans_Larnable_EASAG_ResidualDecoder_AttentionUNet_v2.1'
folds = [4]
save_npz = False
lowres_segmentations = 'None'
part_id = 0
num_parts = 1
num_threads_preprocessing = 6
num_threads_nifti_save = 2
disable_tta = False
overwrite_existing = False
mode = "normal"
all_in_gpu = 'None'
step_size = 0.5
checkpoint_name = 'model_final_checkpoint'
disable_mixed_precision = False

run_inference(
    input_folder= input_folder,
    output_folder= output_folder,
    task=task,
    trainer_class_name = trainer_class_name,
    cascade_trainer_class_name = cascade_trainer_class_name,
    model = model,
    plans_identifier = plans_identifier,
    folds = folds,
    save_npz = save_npz,
    lowres_segmentations = lowres_segmentations,
    part_id = part_id,
    num_parts = num_parts,
    num_threads_preprocessing = num_threads_nifti_save,
    num_threads_nifti_save = num_threads_nifti_save,
    disable_tta = disable_tta,
    overwrite_existing = overwrite_existing,
    mode = mode,
    all_in_gpu = all_in_gpu,
    step_size = step_size,
    checkpoint_name = checkpoint_name,
    disable_mixed_precision = disable_mixed_precision    
)