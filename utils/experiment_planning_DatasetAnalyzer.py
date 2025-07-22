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

from batchgenerators.utilities_file_and_folder_operations import *
from multiprocessing import Pool

from configuration import default_num_threads
from paths import nnUNet_raw_data, nnUNet_cropped_data
import numpy as np
import pickle
from preprocessing_cropping import get_patient_identifiers_from_cropped_files
from skimage.morphology import label
from collections import OrderedDict


class DatasetAnalyzer(object):
    def __init__(self, folder_with_cropped_data, overwrite=True, num_processes=default_num_threads):
        """
        :param folder_with_cropped_data:
        :param overwrite: If True then precomputed values will not be used and instead recomputed from the data.
        False will allow loading of precomputed values. This may be dangerous though if some of the code of this class
        was changed, therefore the default is True.
        """
        print(f"[DEBUG] Initializing DatasetAnalyzer with folder={folder_with_cropped_data}, overwrite={overwrite}, num_processes={num_processes}")
        self.num_processes = num_processes
        self.overwrite = overwrite
        self.folder_with_cropped_data = folder_with_cropped_data
        self.sizes = self.spacings = None
        print(f"[DEBUG] Initialized self.sizes and self.spacings = None")
        self.patient_identifiers = get_patient_identifiers_from_cropped_files(self.folder_with_cropped_data)
        print(f"[DEBUG] Retrieved patient_identifiers: {self.patient_identifiers}")
        assert isfile(join(self.folder_with_cropped_data, "dataset.json")), \
            "dataset.json needs to be in folder_with_cropped_data"
        print(f"[DEBUG] Found dataset.json in {self.folder_with_cropped_data}")
        self.props_per_case_file = join(self.folder_with_cropped_data, "props_per_case.pkl")
        print(f"[DEBUG] Set self.props_per_case_file = {self.props_per_case_file}")
        self.intensityproperties_file = join(self.folder_with_cropped_data, "intensityproperties.pkl")
        print(f"[DEBUG] Set self.intensityproperties_file = {self.intensityproperties_file}")

    def load_properties_of_cropped(self, case_identifier):
        print(f"[DEBUG] load_properties_of_cropped called with case_identifier = {case_identifier}")
        pkl_path = join(self.folder_with_cropped_data, f"{case_identifier}.pkl")
        print(f"[DEBUG] Constructed pkl_path = {pkl_path}")
        with open(pkl_path, 'rb') as f:
            properties = pickle.load(f)
        print(f"[DEBUG] Loaded properties from {pkl_path}: keys = {list(properties.keys())}")
        return properties

    @staticmethod
    def _check_if_all_in_one_region(seg, regions):
        print(f"[DEBUG] _check_if_all_in_one_region called with seg.shape = {seg.shape}, regions = {regions}")
        res = OrderedDict()
        print(f"[DEBUG] Initialized result OrderedDict")
        for r in regions:
            print(f"[DEBUG] Processing region = {r}")
            new_seg = np.zeros(seg.shape)
            print(f"[DEBUG] Created new_seg zeros with shape = {new_seg.shape}")
            for c in r:
                new_seg[seg == c] = 1
                print(f"[DEBUG] Marked class {c} in new_seg")
            labelmap, numlabels = label(new_seg, return_num=True)
            print(f"[DEBUG] Labeled new_seg: numlabels = {numlabels}")
            res[tuple(r)] = (numlabels == 1)
            print(f"[DEBUG] Set res[{tuple(r)}] = {res[tuple(r)]}")
        print(f"[DEBUG] Returning res = {res}")
        return res

    @staticmethod
    def _collect_class_and_region_sizes(seg, all_classes, vol_per_voxel):
        print(f"[DEBUG] _collect_class_and_region_sizes called with seg.shape={seg.shape}, all_classes={all_classes}, vol_per_voxel={vol_per_voxel}")
        volume_per_class = OrderedDict()
        print(f"[DEBUG] Initialized volume_per_class as OrderedDict")
        region_volume_per_class = OrderedDict()
        print(f"[DEBUG] Initialized region_volume_per_class as OrderedDict")
        for c in all_classes:
            print(f"[DEBUG] Processing class {c}")
            region_volume_per_class[c] = []
            print(f"[DEBUG] Set region_volume_per_class[{c}] = []")
            vol_c = np.sum(seg == c) * vol_per_voxel
            volume_per_class[c] = vol_c
            print(f"[DEBUG] Computed volume_per_class[{c}] = {vol_c}")
            labelmap, numregions = label(seg == c, return_num=True)
            print(f"[DEBUG] Labeled seg=={c}, numregions = {numregions}")
            for l in range(1, numregions + 1):
                rv = np.sum(labelmap == l) * vol_per_voxel
                region_volume_per_class[c].append(rv)
                print(f"[DEBUG] Appended region volume for region {l} of class {c}: {rv}")
        print(f"[DEBUG] Returning volume_per_class={volume_per_class}, region_volume_per_class={region_volume_per_class}")
        return volume_per_class, region_volume_per_class

    def _get_unique_labels(self, patient_identifier):
        print(f"[DEBUG] _get_unique_labels called with patient_identifier={patient_identifier}")
        seg = np.load(join(self.folder_with_cropped_data, patient_identifier) + ".npz")['data'][-1]
        print(f"[DEBUG] Loaded seg map shape: {seg.shape}")
        unique_classes = np.unique(seg)
        print(f"[DEBUG] Unique classes found: {unique_classes}")
        return unique_classes

    def _load_seg_analyze_classes(self, patient_identifier, all_classes):
        """
        1) what class is in this training case?
        2) what is the size distribution for each class?
        3) what is the region size of each class?
        4) check if all in one region
        """
        print(f"[DEBUG] _load_seg_analyze_classes called with patient_identifier={patient_identifier}, all_classes={all_classes}")
        seg = np.load(join(self.folder_with_cropped_data, patient_identifier) + ".npz")['data'][-1]
        print(f"[DEBUG] Loaded seg map shape: {seg.shape}")
        pkl = load_pickle(join(self.folder_with_cropped_data, patient_identifier) + ".pkl")
        print(f"[DEBUG] Loaded pickle props keys: {list(pkl.keys())}")
        vol_per_voxel = np.prod(pkl['itk_spacing'])
        print(f"[DEBUG] Computed vol_per_voxel = {vol_per_voxel}")

        # ad 1)
        unique_classes = np.unique(seg)
        print(f"[DEBUG] Found unique_classes = {unique_classes}")

        # 4) check if all in one region
        regions = list()
        regions.append(list(all_classes))
        print(f"[DEBUG] Added full-class region: {regions[-1]}")
        for c in all_classes:
            regions.append((c,))
            print(f"[DEBUG] Added single-class region: {(c,)}")

        all_in_one_region = self._check_if_all_in_one_region(seg, regions)
        print(f"[DEBUG] all_in_one_region = {all_in_one_region}")

        # 2 & 3) region sizes
        volume_per_class, region_sizes = self._collect_class_and_region_sizes(seg, all_classes, vol_per_voxel)
        print(f"[DEBUG] volume_per_class = {volume_per_class}")
        print(f"[DEBUG] region_sizes = {region_sizes}")

        return unique_classes, all_in_one_region, volume_per_class, region_sizes

    def get_classes(self):
        print(f"[DEBUG] get_classes called")
        datasetjson = load_json(join(self.folder_with_cropped_data, "dataset.json"))
        print(f"[DEBUG] Loaded dataset.json labels: {datasetjson.get('labels')}")
        return datasetjson['labels']


    def analyse_segmentations(self):
        print(f"[DEBUG] analyse_segmentations called")
        class_dct = self.get_classes()
        print(f"[DEBUG] Retrieved class_dct: {class_dct}")

        if self.overwrite or not isfile(self.props_per_case_file):
            print(f"[DEBUG] Overwrite is {self.overwrite} or props file missing; computing props_per_patient")
            p = Pool(self.num_processes)
            print(f"[DEBUG] Created Pool with {self.num_processes} processes for _get_unique_labels")
            res = p.map(self._get_unique_labels, self.patient_identifiers)
            print(f"[DEBUG] Mapped _get_unique_labels, result: {res}")
            p.close()
            print(f"[DEBUG] Closed Pool")
            p.join()
            print(f"[DEBUG] Joined Pool")

            props_per_patient = OrderedDict()
            print(f"[DEBUG] Initialized props_per_patient OrderedDict")

            for pid, unique_classes in zip(self.patient_identifiers, res):
                print(f"[DEBUG] Processing patient_identifier={pid}, unique_classes={unique_classes}")
                props = dict()
                props['has_classes'] = unique_classes
                print(f"[DEBUG] Set props['has_classes'] = {unique_classes}")
                props_per_patient[pid] = props
                print(f"[DEBUG] Added entry for {pid} in props_per_patient")

            save_pickle(props_per_patient, self.props_per_case_file)
            print(f"[DEBUG] Saved props_per_patient to {self.props_per_case_file}")
        else:
            print(f"[DEBUG] Loading existing props_per_patient from {self.props_per_case_file}")
            props_per_patient = load_pickle(self.props_per_case_file)
            print(f"[DEBUG] Loaded props_per_patient: {props_per_patient}")

        return class_dct, props_per_patient

    def get_sizes_and_spacings_after_cropping(self):
        print(f"[DEBUG] get_sizes_and_spacings_after_cropping called")
        sizes = []
        print(f"[DEBUG] Initialized sizes list")
        spacings = []
        print(f"[DEBUG] Initialized spacings list")

        for idx, c in enumerate(self.patient_identifiers):
            print(f"[DEBUG] Looping patient_identifiers[{idx}]: {c}")
            properties = self.load_properties_of_cropped(c)
            print(f"[DEBUG] Loaded properties: {properties.keys()}")

            size = properties.get("size_after_cropping")
            sizes.append(size)
            print(f"[DEBUG] Appended size_after_cropping for {c}: {size}")

            spacing = properties.get("original_spacing")
            spacings.append(spacing)
            print(f"[DEBUG] Appended original_spacing for {c}: {spacing}")

        print(f"[DEBUG] Returning sizes: {sizes}, spacings: {spacings}")
        return sizes, spacings


    def get_modalities(self):
        print(f"[DEBUG] get_modalities called")
        datasetjson = load_json(join(self.folder_with_cropped_data, "dataset.json"))
        print(f"[DEBUG] Loaded dataset.json: keys = {list(datasetjson.keys())}")
        modalities = datasetjson["modality"]
        print(f"[DEBUG] Extracted raw modalities dict: {modalities}")
        modalities = {int(k): modalities[k] for k in modalities.keys()}
        print(f"[DEBUG] Converted modality keys to int: {modalities}")
        return modalities

    def get_size_reduction_by_cropping(self):
        print(f"[DEBUG] get_size_reduction_by_cropping called")
        size_reduction = OrderedDict()
        print(f"[DEBUG] Initialized size_reduction OrderedDict")
        for idx, p in enumerate(self.patient_identifiers):
            print(f"[DEBUG] Looping patient_identifiers[{idx}]: {p}")
            props = self.load_properties_of_cropped(p)
            print(f"[DEBUG] Loaded properties for {p}: keys = {list(props.keys())}")
            shape_before_crop = props["original_size_of_raw_data"]
            print(f"[DEBUG] original_size_of_raw_data for {p}: {shape_before_crop}")
            shape_after_crop = props['size_after_cropping']
            print(f"[DEBUG] size_after_cropping for {p}: {shape_after_crop}")
            size_red = np.prod(shape_after_crop) / np.prod(shape_before_crop)
            print(f"[DEBUG] Computed size_reduction for {p}: {size_red}")
            size_reduction[p] = size_red
        print(f"[DEBUG] Returning size_reduction: {size_reduction}")
        return size_reduction

    def _get_voxels_in_foreground(self, patient_identifier, modality_id):
        print(f"[DEBUG] _get_voxels_in_foreground called with patient_identifier={patient_identifier}, modality_id={modality_id}")
        all_data = np.load(join(self.folder_with_cropped_data, patient_identifier) + ".npz")['data']
        print(f"[DEBUG] Loaded all_data shape: {all_data.shape}")
        modality = all_data[modality_id]
        print(f"[DEBUG] Extracted modality data for id {modality_id}, shape: {modality.shape}")
        mask = all_data[-1] > 0
        print(f"[DEBUG] Computed foreground mask, sum(mask) = {int(mask.sum())}")
        voxels = list(modality[mask][::10])  # no need to take every voxel
        print(f"[DEBUG] Sampled voxels count = {len(voxels)}")
        return voxels

    @staticmethod
    def _compute_stats(voxels):
        print(f"[DEBUG] _compute_stats called with voxels length = {len(voxels)}")
        if len(voxels) == 0:
            print(f"[DEBUG] Empty voxels, returning all NaN")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        median = np.median(voxels)
        print(f"[DEBUG] Computed median = {median}")
        mean = np.mean(voxels)
        print(f"[DEBUG] Computed mean = {mean}")
        sd = np.std(voxels)
        print(f"[DEBUG] Computed std = {sd}")
        mn = np.min(voxels)
        print(f"[DEBUG] Computed min = {mn}")
        mx = np.max(voxels)
        print(f"[DEBUG] Computed max = {mx}")
        percentile_99_5 = np.percentile(voxels, 99.5)
        print(f"[DEBUG] Computed 99.5th percentile = {percentile_99_5}")
        percentile_00_5 = np.percentile(voxels, 0.5)
        print(f"[DEBUG] Computed 0.5th percentile = {percentile_00_5}")
        return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5


    def collect_intensity_properties(self, num_modalities):
        print(f"[DEBUG] collect_intensity_properties called with num_modalities={num_modalities}")
        if self.overwrite or not isfile(self.intensityproperties_file):
            print(f"[DEBUG] Overwrite is {self.overwrite} or intensityproperties file missing; computing intensity properties")
            p = Pool(self.num_processes)
            print(f"[DEBUG] Created Pool with {self.num_processes} processes for intensity computation")

            results = OrderedDict()
            print(f"[DEBUG] Initialized results OrderedDict")

            for mod_id in range(num_modalities):
                print(f"[DEBUG] Processing modality {mod_id}")
                results[mod_id] = OrderedDict()
                print(f"[DEBUG] Initialized results[{mod_id}] = OrderedDict")

                v = p.starmap(self._get_voxels_in_foreground, zip(self.patient_identifiers,
                                                                  [mod_id] * len(self.patient_identifiers)))
                print(f"[DEBUG] Retrieved voxel lists for all patients, modality {mod_id}: {v}")

                w = []
                for iv in v:
                    w += iv
                print(f"[DEBUG] Concatenated all sampled voxels into w (length={len(w)})")

                median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = self._compute_stats(w)
                print(f"[DEBUG] Computed global stats for modality {mod_id}: median={median}, mean={mean}, sd={sd}, mn={mn}, mx={mx}, 99.5%={percentile_99_5}, 0.5%={percentile_00_5}")

                local_props = p.map(self._compute_stats, v)
                print(f"[DEBUG] Computed local stats for each patient: {local_props}")

                props_per_case = OrderedDict()
                print(f"[DEBUG] Initialized props_per_case OrderedDict")

                for i, pat in enumerate(self.patient_identifiers):
                    print(f"[DEBUG] Setting props for patient {pat}")
                    props_per_case[pat] = OrderedDict()
                    props_per_case[pat]['median'] = local_props[i][0]
                    props_per_case[pat]['mean'] = local_props[i][1]
                    props_per_case[pat]['sd'] = local_props[i][2]
                    props_per_case[pat]['mn'] = local_props[i][3]
                    props_per_case[pat]['mx'] = local_props[i][4]
                    props_per_case[pat]['percentile_99_5'] = local_props[i][5]
                    props_per_case[pat]['percentile_00_5'] = local_props[i][6]
                    print(f"[DEBUG] props_per_case[{pat}] = {props_per_case[pat]}")

                results[mod_id]['local_props'] = props_per_case
                results[mod_id]['median'] = median
                results[mod_id]['mean'] = mean
                results[mod_id]['sd'] = sd
                results[mod_id]['mn'] = mn
                results[mod_id]['mx'] = mx
                results[mod_id]['percentile_99_5'] = percentile_99_5
                results[mod_id]['percentile_00_5'] = percentile_00_5
                print(f"[DEBUG] Set results[{mod_id}] stats: {results[mod_id]}")

            p.close()
            print(f"[DEBUG] Closed Pool for intensity computation")
            p.join()
            print(f"[DEBUG] Joined Pool for intensity computation")

            save_pickle(results, self.intensityproperties_file)
            print(f"[DEBUG] Saved intensityproperties to {self.intensityproperties_file}")
        else:
            print(f"[DEBUG] Loading existing intensityproperties from {self.intensityproperties_file}")
            results = load_pickle(self.intensityproperties_file)
            print(f"[DEBUG] Loaded intensityproperties: {results}")

        return results

    def analyze_dataset(self, collect_intensityproperties=True):
        print(f"[DEBUG] analyze_dataset called with collect_intensityproperties={collect_intensityproperties}")

        # get all spacings and sizes
        sizes, spacings = self.get_sizes_and_spacings_after_cropping()
        print(f"[DEBUG] Retrieved sizes: {sizes}")
        print(f"[DEBUG] Retrieved spacings: {spacings}")

        # get all classes
        classes = self.get_classes()
        print(f"[DEBUG] Retrieved classes: {classes}")
        all_classes = [int(i) for i in classes.keys() if int(i) > 0]
        print(f"[DEBUG] all_classes (excluding background): {all_classes}")

        # modalities
        modalities = self.get_modalities()
        print(f"[DEBUG] Retrieved modalities: {modalities}")

        # collect intensity information
        if collect_intensityproperties:
            print(f"[DEBUG] Collecting intensityproperties")
            intensityproperties = self.collect_intensity_properties(len(modalities))
            print(f"[DEBUG] intensityproperties: {intensityproperties}")
        else:
            intensityproperties = None
            print(f"[DEBUG] Skipped intensityproperties collection")

        # size reduction by cropping
        size_reductions = self.get_size_reduction_by_cropping()
        print(f"[DEBUG] size_reductions: {size_reductions}")

        dataset_properties = dict()
        dataset_properties['all_sizes'] = sizes
        dataset_properties['all_spacings'] = spacings
        dataset_properties['all_classes'] = all_classes
        dataset_properties['modalities'] = modalities
        dataset_properties['intensityproperties'] = intensityproperties
        dataset_properties['size_reductions'] = size_reductions
        print(f"[DEBUG] Assembled dataset_properties: {dataset_properties}")

        save_pickle(dataset_properties, join(self.folder_with_cropped_data, "dataset_properties.pkl"))
        print(f"[DEBUG] Saved dataset_properties to dataset_properties.pkl")

        return dataset_properties

