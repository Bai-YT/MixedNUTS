import torch
from torch import nn
from robustbench import load_model

from os.path import join
import numpy as np

from models.base_models import bit_rn, convnext_v2
from models.nonlin_mixed_classifier import NonLinMixedClassifier


def load_sd_relaxed(model, state_dict):
    """ Load state_dict for a model and ignore expected harmless exceptions. """

    try:  # Load state_dict
        return model.load_state_dict(state_dict)

    except Exception as e:
        allowed_exceptions = [
            'Missing key(s) in state_dict: "module.mean", "module.std".',
            'Missing key(s) in state_dict: "mean", "std".'
        ]
        for allowed_exc in allowed_exceptions:
            if allowed_exc in str(e):
                return model.load_state_dict(state_dict, strict=False)
        raise e


def load_std_model(
    std_model_arch, std_load_path, num_classes=10,
    return_features=False, parallelize=False, device=torch.device('cpu')
):
    """ Load standard base classifier. """
    print(f"Loading standard model: {std_model_arch} from {std_load_path}...")

    if std_model_arch == 'convnext_v2-l_224':
        # Load ConvNext V2 model
        assert num_classes == 1000
        std_model = convnext_v2.convnextv2_large(num_classes=num_classes)
        sd = torch.load(std_load_path, map_location='cpu')
        load_sd_relaxed(std_model, sd["model"])

    elif std_model_arch == 'rn152':
        # Load BiT ResNet model
        std_model = bit_rn.KNOWN_MODELS["BiT-M-R152x2"](
            head_size=num_classes, zero_head=False, return_features=return_features
        )

        try:  # Load model checkpoint from disk
            std_model.load_from(np.load(std_load_path))

        except:  # Perhaps the state_dict corresponds to a parallelized model
            state_dict = torch.load(std_load_path, map_location='cpu')
            if 'model' in state_dict.keys():  # Remove redundant keys
                state_dict = state_dict['model']
            std_model = nn.DataParallel(std_model)  # Wrap model in DataParallel
            load_sd_relaxed(std_model, state_dict)
            std_model = std_model.module  # Remove DataParallel wrapper

    else:
        raise ValueError(f"Unknown standard model architecture: {std_model_arch}")

    # Freeze models, set to eval mode, and parallelize
    std_model.eval()
    for param in std_model.parameters():
        param.requires_grad = False

    # Move mode to device
    if parallelize and device == torch.device('cuda'):
        return nn.DataParallel(std_model)
    else:
        return std_model.to(device)


def load_rob_model(
    rob_model_name, dataset_name, pgd_type='l_inf', parallelize=False, device=torch.device('cpu')
):
    """ Load robust base classifier. """

    if rob_model_name is not None:  # Load RobustBench model
        print(f"Loading robust model: {rob_model_name} from RobustBench...")
        rob_model = load_model(
            rob_model_name, model_dir='./base_models', dataset=dataset_name,
            threat_model=('Linf' if pgd_type == 'l_inf' else 'L2')
        )
    else:
        return None

    # Freeze models, set to eval mode, and parallelize
    rob_model.eval()
    for param in rob_model.parameters():
        param.requires_grad = False

    if parallelize and device == torch.device('cuda'):
        return nn.DataParallel(rob_model)
    else:
        return rob_model.to(device)


def get_nonlin_mixed_classifier(forward_settings, dataset_name, root_dir):
    """ This function is used to build the mixed classifier for RobustBench. """

    num_classes = {"cifar10": 10, "cifar100": 100, "imagenet": 1000}[dataset_name]

    # Load robust base model
    rob_model_name = forward_settings["rob_model_name"]
    rob_model = load_rob_model(rob_model_name, dataset_name)

    # Load standard base model
    std_model_arch = forward_settings["std_model_arch"]
    std_load_path = join(root_dir, dataset_name, f"{dataset_name}_std_{std_model_arch}.pt")
    std_model = load_std_model(std_model_arch, std_load_path, num_classes)

    # Mixed classifier
    mix_model = NonLinMixedClassifier(std_model, rob_model, forward_settings)

    return mix_model.to(torch.device(
        'cuda:0' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    ))
