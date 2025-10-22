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

import shutil
from copy import deepcopy

from inference_segmentation_export import save_segmentation_nifti_from_softmax
from batchgenerators.utilities_file_and_folder_operations import *
import numpy as np
from multiprocessing import Pool
from postprocessing_connected_components import apply_postprocessing_to_folder, load_postprocessing


def merge_files(files, properties_files, out_file, override, store_npz):
    if override or not isfile(out_file):
        # Load softmax arrays
        softmax_list = []
        for f in files:
            data = np.load(f)
            sm = data['softmax'][None]
            softmax_list.append(sm)
        softmax = np.vstack(softmax_list)
        softmax = np.mean(softmax, 0)

        # Load properties files
        props = []
        for pf in properties_files:
            p = load_pickle(pf)
            props.append(p)

        # Determine region class orders
        reg_class_orders = []
        for p in props:
            rco = p.get('regions_class_order', None)
            reg_class_orders.append(rco)

        if not all([i is None for i in reg_class_orders]):
            tmp = reg_class_orders[0]
            for r in reg_class_orders[1:]:
                assert tmp == r, (
                    f"If merging files with regions_class_order, the regions_class_orders of all files must be the same."
                    f" Found: {reg_class_orders}, files: {files}"
                )
            regions_class_order = tmp
        else:
            regions_class_order = None

        # Save segmentation NIfTI
        save_segmentation_nifti_from_softmax(
            softmax, out_file, props[0], 3,
            regions_class_order, None, None, force_separate_z=None
        )
    
        if store_npz:
            npz_out = out_file[:-7] + ".npz"
            pkl_out = out_file[:-7] + ".pkl"
            np.savez_compressed(npz_out, softmax=softmax)
            save_pickle(props, pkl_out)
    else:
        print(f"Skipping merge: out_file exists ({out_file}) and override is False")



def merge(folders, output_folder, threads, override=True, postprocessing_file=None, store_npz=False):

    maybe_mkdir_p(output_folder)
    if postprocessing_file is not None:
        output_folder_orig = deepcopy(output_folder)
        output_folder = join(output_folder, 'not_postprocessed')
        maybe_mkdir_p(output_folder)
    else:
        output_folder_orig = None

    patient_ids = [subfiles(i, suffix=".npz", join=False) for i in folders]
    patient_ids = [i for j in patient_ids for i in j]
    patient_ids = [i[:-4] for i in patient_ids]
    patient_ids = np.unique(patient_ids)

    for f in folders:
        missing_npz = [i for i in patient_ids if not isfile(join(f, i + ".npz"))]
        assert not missing_npz, f"Not all patient npz are available in {f}"
        missing_pkl = [i for i in patient_ids if not isfile(join(f, i + ".pkl"))]
        assert not missing_pkl, f"Not all patient pkl are available in {f}"

    files = []
    property_files = []
    out_files = []
    for p in patient_ids:
        f_list = [join(f, p + ".npz") for f in folders]
        pf_list = [join(f, p + ".pkl") for f in folders]
        out_path = join(output_folder, p + ".nii.gz")
        files.append(f_list)
        property_files.append(pf_list)
        out_files.append(out_path)
    
    p = Pool(threads)
    p.starmap(merge_files, zip(files, property_files, out_files, [override] * len(out_files), [store_npz] * len(out_files)))
    p.close()
    p.join()

def inference_ensemble_merge_predictions(
    folders: list[str],
    output_folder: str,
    threads: int = 2,
    postprocessing_file: str | None = None,
    store_npz: bool = False
):
    """
    Merge multiple nnU-Net prediction folders containing .npz files into nifti outputs.

    Parameters:
    - folders: List of paths to folders with .npz prediction files.
    - output_folder: Path where merged nifti results will be saved.
    - threads: Number of threads for writing nifti files. Default: 2.
    - postprocessing_file: Path to postprocessing configuration file. If None, no postprocessing is applied.
    - store_npz: If True, also store .npz and .pkl outputs. Default: False.

    Returns:
    None
    """
    merge(
        folders,
        output_folder,
        threads,
        override=True,
        postprocessing_file=postprocessing_file,
        store_npz=store_npz
    )
