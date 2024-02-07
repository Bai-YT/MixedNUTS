import torch
from torchvision import transforms
import click
import yaml

from models.output_maps import IdentityMap, HardMaxMap, ScaleMap, LNClampPowerScaleMap
from utils.model_utils import get_nonlin_mixed_classifier
from utils.robust_bench import benchmark
from utils.misc_utils import seed_all

SEED = 20230331


@click.command(context_settings={'show_default': True})
@click.option(
    '--root_dir', default=".", show_default=True,
    help="Path to the root directory that stores the models"
)
@click.option(
    '--dataset_name', required=True,
    type=click.Choice(['cifar10', 'cifar100', 'imagenet']),
    help="The dataset to experiment with. One of {'cifar10', 'cifar100', 'imagenet'}."
)
@click.option(
    '--rob_model_name', type=str, required=True,
    help="The robust base model to use. Options include "
         "{'Gowal2020Uncovering_extra', 'Wang2023Better_WRN-70-16', 'Peng2023Robust'}."
)
@click.option(  # Change for public version
    '--std_model_arch', required=True,
    type=click.Choice(['rn152', 'convnext_v2-l_224']),
    help="Standard model architecture. One of {'rn152', 'convnext_v2-l_224'}."
)
@click.option(
    '--map_type', type=click.Choice(['identity', 'best']), required=True,
    help="Output mapping combination (one of {'identity', 'best'})."
)
@click.option(
    '--beta', type=float, default=None, show_default=True,
    help="The beta value to be used. If None, use the default value stored in "
         "optimal_spca.yaml. Default to None."
)
@click.option(
    '--ln_k', type=int, default=250, show_default=True,
    help="The top-k option for layer norm. Default to 250."
)
@click.option(
    '--use_fp16/--use_fp32', default=False, show_default=True,
    help="Use mixed precision (fp16) or not (fp32)."
)
@click.option(
    '--adaptive/--default', default=False, show_default=True,
    help=("If true, run the adaptive AutoAttack, "
          "which works better for the nonlinear mixed classifiers")
)
@click.option(
    '--n_examples', default=10000, show_default=True,
    help="Number of evaluation examples. Default to 10000."
)
@click.option(
    '--batch_size_per_gpu', default=40, show_default=True,
    help="Batch size per GPU. Default to 40."
)
@click.option(
    '--use_nonlin_for_grad/--disable_nonlin_for_grad', default=True, show_default=True,
    help="Whether to include the robust base model nonlinearity for gradient."
)
@click.option(
    '--device', default=(
        'cuda' if torch.cuda.is_available() else 
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
)

def run_robustbench(
    dataset_name, root_dir, rob_model_name, std_model_arch, map_type, beta, ln_k,
    use_fp16, adaptive, n_examples, batch_size_per_gpu, use_nonlin_for_grad, device
):
    """ Performs robustbench evaluation. """

    model_full_name = f"mixednuts_{rob_model_name}"
    threat_model = "Linf"  # one of {"Linf", "L2", "corruptions"}

    # Default transformation hyperparameters
    with open('configs/optimal_spca.yaml', 'r') as file:
        dflt_dic = yaml.safe_load(file)[dataset_name][rob_model_name]

    beta_diffable = beta if beta is not None else dflt_dic['none']['default_beta']
    alpha_diffable = dflt_dic['none'][beta_diffable]['alpha']

    if map_type == 'identity':  # Without nonlinear transformation
        beta = beta if beta is not None else dflt_dic['none']['default_beta']
        alpha = dflt_dic['none'][beta]['alpha']
        std_map, rob_map = IdentityMap(), IdentityMap()

    elif map_type == 'best':    # With best nonlinear transformation
        std_map = HardMaxMap()
        beta = beta if beta is not None else dflt_dic['gelu']['default_beta']
        nonlin_spca = dflt_dic['gelu'][beta]
        rob_map = LNClampPowerScaleMap(
            scale=nonlin_spca['s'], power=nonlin_spca['p'],
            clamp_bias=nonlin_spca['c'], ln_k=ln_k
        )
        alpha = nonlin_spca['alpha']

    # Consolidate into forward_settings dict
    forward_settings = {
        "std_model_arch": std_model_arch, "rob_model_name": rob_model_name,
        "std_map": std_map, "rob_map": rob_map, "use_nonlin_for_grad": use_nonlin_for_grad,
        "alpha": alpha, "alpha_diffable": alpha_diffable,
        "parallel": True, "enable_autocast": use_fp16
    }

    # Build NonLinearMixedClassifier
    mix_model = get_nonlin_mixed_classifier(
        forward_settings=forward_settings, dataset_name=dataset_name, root_dir=root_dir
    )
    mix_model = torch.nn.DataParallel(mix_model)
    mix_model.eval()
    for param in mix_model.parameters():
        param.requires_grad = False

    if dataset_name == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor()
        ])
    else:  # Transformation must be None for CIFAR
        transform = None

    eps = 4. / 255. if dataset_name == 'imagenet' else 8. / 255.
    print(f"RobustBench on {dataset_name} with eps={eps:.4f} {threat_model} attack.")
    batch_size = batch_size_per_gpu * max(torch.cuda.device_count(), 1)

    # Run RobustBench benchmark!
    seed_all(SEED)
    clean_acc, attacked_acc = benchmark(
        mix_model, model_name=model_full_name, to_disk=True, threat_model=threat_model,
        dataset=dataset_name, data_dir=f"data/{dataset_name}", n_examples=n_examples,
        batch_size=min(batch_size, n_examples), eps=eps, adaptive=adaptive,
        preprocessing=transform, device=torch.device(device)
    )
    print(f"Clean accuracy: {clean_acc:.2%}.")
    print(f"{'Adaptive ' if adaptive else ''}AutoAttacked accuracy: {attacked_acc:.2%}.")


if __name__ == "__main__":
    run_robustbench()
