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





def load_pretrained_weights(network, fname, verbose=False):
    """
    THIS DOES NOT TRANSFER SEGMENTATION HEADS!
    """
    print(f"[DEBUG] loading pretrained weights: {fname}, verbose={verbose}")
    saved_model = torch.load(fname)
    print(f"[DEBUG] model loaded from disk")
    pretrained_dict = saved_model['state_dict']
    print(f"[DEBUG] extracted state_dict with {len(pretrained_dict)} keys")

    new_state_dict = {}

    # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in pretrained_dict.items():
        print(f"[DEBUG] processing key: {k}")
        key = k
        # remove module. prefix from DDP models
        if key.startswith('module.'):
            key = key[7:]
            print(f"[DEBUG] stripped 'module.' prefix, new key: {key}")
        new_state_dict[key] = value
    print(f"[DEBUG] new_state_dict populated with {len(new_state_dict)} keys")

    pretrained_dict = new_state_dict

    model_dict = network.state_dict()
    print(f"[DEBUG] current network state_dict has {len(model_dict)} keys")
    ok = True
    for key, _ in model_dict.items():
        print(f"[DEBUG] checking model key: {key}")
        if ('conv_blocks' in key):
            if (key in pretrained_dict) and (model_dict[key].shape == pretrained_dict[key].shape):
                print(f"[DEBUG] key matches and shape matches: {key}")
                continue
            else:
                print(f"[DEBUG] key mismatch or shape mismatch: {key}")
                ok = False
                break

    # filter unnecessary keys
    if ok:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        print(f"[DEBUG] filtered pretrained_dict to {len(pretrained_dict)} compatible keys")
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        print("################### Loading pretrained weights from file ", fname, '###################')
        if verbose:
            print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
            for key, _ in pretrained_dict.items():
                print(key)
        print("################### Done ###################")
        network.load_state_dict(model_dict)
        print(f"[DEBUG] network.load_state_dict executed")
    else:
        print(f"[DEBUG] pretrained weights compatibility check failed, raising RuntimeError")
        raise RuntimeError("Pretrained weights are not compatible with the current network architecture")

