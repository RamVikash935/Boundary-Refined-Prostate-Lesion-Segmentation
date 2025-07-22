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
    print("Entering merge_files")
    print(f"Parameters - files: {files}, properties_files: {properties_files}, out_file: {out_file}, override: {override}, store_npz: {store_npz}")

    if override or not isfile(out_file):
        print(f"Override is {override} or out_file does not exist: {not isfile(out_file)}")

        # Load softmax arrays
        softmax_list = []
        for f in files:
            print(f"Loading softmax from file: {f}")
            data = np.load(f)
            print(f"Loaded keys from npz: {list(data.keys())}")
            sm = data['softmax'][None]
            print(f"softmax shape with new axis: {sm.shape}")
            softmax_list.append(sm)
        softmax = np.vstack(softmax_list)
        print(f"Stacked softmax shape: {softmax.shape}")
        softmax = np.mean(softmax, 0)
        print(f"Averaged softmax shape: {softmax.shape}")

        # Load properties files
        props = []
        for pf in properties_files:
            print(f"Loading properties pickle from: {pf}")
            p = load_pickle(pf)
            print(f"Loaded properties keys: {list(p.keys())}")
            props.append(p)

        # Determine region class orders
        reg_class_orders = []
        for p in props:
            rco = p.get('regions_class_order', None)
            print(f"regions_class_order for properties: {rco}")
            reg_class_orders.append(rco)

        if not all([i is None for i in reg_class_orders]):
            print("Non-None region class orders found, verifying consistency")
            tmp = reg_class_orders[0]
            for r in reg_class_orders[1:]:
                print(f"Comparing {tmp} to {r}")
                assert tmp == r, (
                    f"If merging files with regions_class_order, the regions_class_orders of all files must be the same."
                    f" Found: {reg_class_orders}, files: {files}"
                )
            regions_class_order = tmp
            print(f"Using regions_class_order: {regions_class_order}")
        else:
            regions_class_order = None
            print("All region class orders are None, setting regions_class_order to None")

        # Save segmentation NIfTI
        print(f"Saving segmentation NIfTI to {out_file}")
        save_segmentation_nifti_from_softmax(
            softmax, out_file, props[0], 3,
            regions_class_order, None, None, force_separate_z=None
        )
        print("Saved NIfTI segmentation")

        if store_npz:
            npz_out = out_file[:-7] + ".npz"
            pkl_out = out_file[:-7] + ".pkl"
            print(f"store_npz is True, saving compressed npz to {npz_out}")
            np.savez_compressed(npz_out, softmax=softmax)
            print(f"Saved compressed npz: {npz_out}")
            print(f"Saving properties pickle to {pkl_out}")
            save_pickle(props, pkl_out)
            print(f"Saved properties pickle: {pkl_out}")
    else:
        print(f"Skipping merge: out_file exists ({out_file}) and override is False")



def merge(folders, output_folder, threads, override=True, postprocessing_file=None, store_npz=False):
    print("Entering merge")
    print(f"Parameters - folders: {folders}, output_folder: {output_folder}, threads: {threads}, override: {override}, postprocessing_file: {postprocessing_file}, store_npz: {store_npz}")

    print(f"Ensuring output_folder exists: {output_folder}")
    maybe_mkdir_p(output_folder)
    print(f"Created or confirmed directory: {output_folder}")

    if postprocessing_file is not None:
        print(f"postprocessing_file provided: {postprocessing_file}")
        output_folder_orig = deepcopy(output_folder)
        print(f"Saved original output_folder: {output_folder_orig}")
        output_folder = join(output_folder, 'not_postprocessed')
        print(f"Switched output_folder to: {output_folder}")
        maybe_mkdir_p(output_folder)
        print(f"Created or confirmed directory: {output_folder}")
    else:
        print("No postprocessing_file provided, skipping postprocessed subfolder")
        output_folder_orig = None

    print("Collecting patient IDs from folders")
    patient_ids = [subfiles(i, suffix=".npz", join=False) for i in folders]
    print(f"Raw patient_ids lists: {patient_ids}")
    patient_ids = [i for j in patient_ids for i in j]
    print(f"Flattened patient_ids: {patient_ids}")
    patient_ids = [i[:-4] for i in patient_ids]
    print(f"Stripped extensions from patient_ids: {patient_ids}")
    patient_ids = np.unique(patient_ids)
    print(f"Unique patient_ids: {patient_ids}")

    print("Verifying all .npz and .pkl files exist for each patient across folders")
    for f in folders:
        missing_npz = [i for i in patient_ids if not isfile(join(f, i + ".npz"))]
        print(f"Missing .npz in {f}: {missing_npz}")
        assert not missing_npz, f"Not all patient npz are available in {f}"
        missing_pkl = [i for i in patient_ids if not isfile(join(f, i + ".pkl"))]
        print(f"Missing .pkl in {f}: {missing_pkl}")
        assert not missing_pkl, f"Not all patient pkl are available in {f}"
    print("All required files verified.")

    print("Preparing file lists for merging")
    files = []
    property_files = []
    out_files = []
    for p in patient_ids:
        f_list = [join(f, p + ".npz") for f in folders]
        pf_list = [join(f, p + ".pkl") for f in folders]
        out_path = join(output_folder, p + ".nii.gz")
        print(f"Patient {p}: npz files: {f_list}")
        print(f"Patient {p}: pkl files: {pf_list}")
        print(f"Patient {p}: output file: {out_path}")
        files.append(f_list)
        property_files.append(pf_list)
        out_files.append(out_path)
    print(f"Files prepared for all patients. Total: {len(out_files)}")

    print(f"Starting multiprocessing pool with {threads} threads")
    p = Pool(threads)
    print("Mapping merge_files across file sets")
    p.starmap(merge_files, zip(files, property_files, out_files, [override] * len(out_files), [store_npz] * len(out_files)))
    print("Tasks dispatched to pool")
    p.close()
    print("Closed pool to new tasks")
    p.join()
    print("All merge tasks completed and pool joined.")


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
