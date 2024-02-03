import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union, Dict

import torch
from torch import nn

from adaptive_autoattack import AdaptiveAutoAttack
from autoattack import AutoAttack
from autoattack.state import EvaluationState

from robustbench.eval import corruptions_evaluation
from robustbench.data import get_preprocessing
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import clean_accuracy, update_json
from utils.data_utils import load_clean_dataset


CORRUPTIONS = (
    "shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise",
    "defocus_blur", "brightness", "fog", "zoom_blur", "frost", "glass_blur",
    "impulse_noise", "contrast", "jpeg_compression", "elastic_transform"
)

CORRUPTIONS_3DCC = (
    'near_focus', 'far_focus', 'bit_error', 'color_quant', 'flash', 'fog_3d',
    'h265_abr', 'h265_crf', 'iso_noise', 'low_light', 'xy_motion_blur', 'z_motion_blur'
)

CORRUPTIONS_DICT: Dict[BenchmarkDataset, Tuple[str, ...]] = {
    BenchmarkDataset.cifar_10: {"corruptions": CORRUPTIONS},
    BenchmarkDataset.cifar_100: {"corruptions": CORRUPTIONS},
    BenchmarkDataset.imagenet: {"corruptions": CORRUPTIONS, 
                                "corruptions_3d": CORRUPTIONS_3DCC}
}


def benchmark(
    model: Union[nn.Module, Sequence[nn.Module]],
    n_examples: int = 10000,
    adaptive: bool = False,
    dataset: Union[str, BenchmarkDataset] = BenchmarkDataset.cifar_10,
    threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
    to_disk: bool = False,
    model_name: Optional[str] = None,
    data_dir: str = "./data",
    corruptions_data_dir: Optional[str] = None,
    device: Optional[Union[torch.device, Sequence[torch.device]]] = None,
    batch_size: int = 32,
    eps: Optional[float] = None,
    log_path: Optional[str] = None,
    preprocessing: Optional[Union[str, Callable]] = None,
    aa_state_path: Optional[Path] = None) -> Tuple[float, float]:
    """Benchmarks the given model(s).

    It is possible to benchmark on three threat models and save the results on disk.
    In the future benchmarking multiple models in parallel is going to be possible.

    :param model:           The model to benchmark.
    :param n_examples:      The number of examples to use to benchmark the model.
    :param adaptive:        If True, run the adaptive AutoAttack, 
                            which works with nonlinear mixed classifiers.
    :param dataset:         The dataset to use to benchmark. 
                            Must be one of {cifar10, cifar100}.
    :param threat_model:    The threat model to use to benchmark, 
                            must be one of {L2, Linf, corruptions}.
    :param to_disk:         Whether the results must be saved on disk as .json.
    :param model_name:      The name of the model to use to save the results. 
                            Must be specified if to_json is True.
    :param data_dir:        The directory where the dataset is or 
                            where the dataset should be downloaded.
    :param device:          The device to run the computations.
    :param batch_size:      The batch size to run the computations. 
                            The larger, the faster the evaluation is.
    :param eps:             The epsilon to use for L2 and Linf threat models. 
                            Must not be specified for corruptions threat model.
    :param preprocessing:   The preprocessing that should be used for ImageNet benchmarking.
                            Should be specified if `dataset` is `imageget`.
    :param aa_state_path:   The path where the AA state will be saved and from where should
                            be loaded if it already exists. If `None` no state will be used.

    :return: A Tuple with the clean accuracy and the accuracy in the given threat model.
    """
    if isinstance(model, Sequence) or isinstance(device, Sequence):
        # Multiple models evaluation in parallel not yet implemented
        raise NotImplementedError

    try:
        if model.training:
            warnings.warn(Warning("The given model is *not* in eval mode."))
    except AttributeError:
        warnings.warn(
            Warning(
                "It is not possible to asses if the model is in eval mode"))

    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    threat_model_: ThreatModel = ThreatModel(threat_model)

    device = device or torch.device("cpu")
    model = model.to(device)

    if isinstance(dataset, str) and 'cifar' in dataset.lower():
        print(f"Using the {dataset} dataset.")
        assert preprocessing is None, "preprocessing must be None for CIFAR-10 or 100."
    prepr = get_preprocessing(dataset_, threat_model_, model_name, preprocessing)
    clean_x_test, clean_y_test = load_clean_dataset(dataset_, n_examples, data_dir, prepr)

    accuracy = clean_accuracy(
        model, clean_x_test, clean_y_test, batch_size=batch_size, device=device
    )
    print(f'Clean accuracy: {accuracy:.2%}')

    extra_metrics = {}  # dict to store corruptions_mce for corruptions threat models
    if threat_model_ in {ThreatModel.Linf, ThreatModel.L2}:
        assert eps is not None, "If threat model is L2 or Linf, eps must be specified."

        if adaptive:
            adversary = AdaptiveAutoAttack(
                model, norm=threat_model_.value, eps=eps, version='standard',
                device=device, log_path=log_path
            )
        else:
            adversary = AutoAttack(
                model, norm=threat_model_.value, eps=eps, version='standard',
                device=device, log_path=log_path
            )

        x_adv = adversary.run_standard_evaluation(
            clean_x_test, clean_y_test, bs=batch_size, state_path=aa_state_path
        )
        if aa_state_path is None:
            adv_accuracy = clean_accuracy(
                model, x_adv, clean_y_test, batch_size=batch_size, device=device
            )
        else:
            aa_state = EvaluationState.from_disk(aa_state_path)
            assert aa_state.robust_flags is not None
            adv_accuracy = aa_state.robust_flags.mean().item()
    
    elif threat_model_ in [ThreatModel.corruptions, ThreatModel.corruptions_3d]:
        corruptions = CORRUPTIONS_DICT[dataset_][threat_model_]
        print(f"Evaluating over {len(corruptions)} corruptions")
        # Exceptionally, for corruptions (2d and 3d) we use only resizing to 224x224
        prepr = get_preprocessing(dataset_, threat_model_, model_name, 'Res224')
        # Save into a dict to make a Pandas DF with nested index        
        corruptions_data_dir = corruptions_data_dir or data_dir

        adv_accuracy, extra_metrics['corruptions_mce'] = corruptions_evaluation(
            batch_size, corruptions_data_dir, dataset_, threat_model_, 
            device, model, n_examples, to_disk, prepr, model_name)

    else:
        raise NotImplementedError

    print(f'Adversarial accuracy: {adv_accuracy:.2%}')

    if to_disk:
        assert model_name is not None, \
            "If to_disk is True, model_name must be specified."
        update_json(
            dataset=dataset_, threat_model=threat_model_, model_name=model_name,
            accuracy=accuracy, adv_accuracy=adv_accuracy, eps=eps
        )

    return accuracy, adv_accuracy
