import click
import torch
from os import makedirs
import yaml

from models.output_maps import MappedModel, \
    IdentityMap, LayerNormMap, HardMaxMap, ScaleMap, LNPowerScaleMap, LNClampPowerScaleMap

from utils.margin_utils import get_autoattacked_images, get_pgd20_attacked_images, \
    get_models_and_loader, margin_single_model, save_margin
from utils.misc_utils import seed_all

SEED = 20230331


def _prepare_margin(
    dataset_name, model_name, image_type, pgd_type, n_examples, batch_size, device
):
    model_par_dir, dataset_dir = "base_models", f"data/{dataset_name}"
    assert n_examples <= (
        10000 if 'cifar' in dataset_name else 50000 if image_type == 'clean' else 5000
    ), "n_examples must not exceed 5000 for ImageNet or 10000 for CIFAR."

    # Prepare the STD and ROB models as well as the dataloader.
    get_attacked_images_fn = (
        get_autoattacked_images if image_type == "aa" 
        else get_pgd20_attacked_images if image_type == "pgd20"
        else None
    )
    std_model, rob_model, test_loader = get_models_and_loader(
        dataset_name=dataset_name, model_par_dir=model_par_dir, device=device,
        rob_model_name=(None if model_name == 'std' else model_name),
        dataset_dir=dataset_dir, pgd_type=pgd_type, batch_size=batch_size, n_examples=n_examples
    )
    return std_model, rob_model, test_loader, get_attacked_images_fn


def get_margin_single(
    dataset_name, model_name, use_rob_map, no_ln,
    image_type, pgd_type, n_examples, batch_size, ln_k, device
):
    """ Calculate accuracy and margins of a single model.
    """
    std_model, rob_model, test_loader, get_attacked_images_fn = _prepare_margin(
        dataset_name, model_name, image_type, pgd_type, n_examples, batch_size, device
    )
    model = std_model if model_name == 'std' else rob_model

    if model_name != 'std' and use_rob_map:
        print(f"Using optimal nonlinear transformation for {model_name}.")
        with open('configs/optimal_spca.yaml', 'r') as file:
            dflt_dic = yaml.safe_load(file)[dataset_name][model_name]
        nonlin_spca = dflt_dic['gelu'][dflt_dic['gelu']['default_beta']]
        model = MappedModel(model, LNClampPowerScaleMap(
            scale=nonlin_spca['s'], power=nonlin_spca['p'], clamp_bias=nonlin_spca['c'], ln_k=ln_k
        )).to(device)
    elif no_ln:  # If no_ln is selected, do not use layer norm
        model = MappedModel(model, IdentityMap()).to(device)
    else:        # Otherwise, use layer norm
        model = MappedModel(model, LayerNormMap(ln_k=ln_k)).to(device)
    model = torch.nn.DataParallel(model)

    epsilon = .5 if pgd_type == 'l_2' else 4. / 255. if dataset_name == 'imagenet' else 8. / 255.
    acc, save_dic = margin_single_model(
        model=model, dataset_name=dataset_name, test_loader=test_loader, batch_size=batch_size,
        get_attacked_images_fn=get_attacked_images_fn, pgd_type=pgd_type,
        epsilon=epsilon, n_examples=n_examples, device=device, seed=SEED
    )
    appendix = '' if no_ln else '_map' if use_rob_map else '_ln'
    dir_name = f"{dataset_name}/{model_name}/{image_type}{appendix}"
    makedirs(f"results/{dir_name}", exist_ok=True)
    save_name = f"{dir_name}/{dataset_name}_{model_name}_{image_type}{appendix}"

    print(f"\nAccuracy of the {model_name} model on "
          f"{n_examples} {image_type} test images: {acc:.3f} %.")
    with open(f"results/{save_name}_summary.csv", 'w') as summ_file:
        summ_file.write(
            f"Accuracy of the {model_name} model on "
            f"{n_examples} {image_type} test images: {acc:.3f} %.\n"
        )

    save_margin(save_dic, save_name, n_examples, save_all=(n_examples <= 2000))


def get_margin_pair(
    dataset_name, rob_model_name, use_rob_map, no_ln,
    image_type, pgd_type, n_examples, batch_size, ln_k, device
):
    """ Calculate accuracy and margins of a pair of models, one of which is the
        standard base classifier and the other is the specified robust base model.
    """
    std_model, rob_model, test_loader, get_attacked_images_fn = _prepare_margin(
        dataset_name, rob_model_name, image_type, pgd_type, n_examples, batch_size, device
    )
    # Apply nonlinear mappings to the base classifiers
    std_map = LayerNormMap(ln_k=ln_k) if not no_ln else IdentityMap()
    std_model = MappedModel(std_model, std_map).to(device)

    if use_rob_map:
        print(f"Using optimal nonlinear transformation for {rob_model_name}.")
        with open('configs/optimal_spca.yaml', 'r') as file:
            dflt_dic = yaml.safe_load(file)[dataset_name][rob_model_name]
        nonlin_spca = dflt_dic['gelu']
        rob_model = MappedModel(rob_model, LNClampPowerScaleMap(
            scale=nonlin_spca['s'], power=nonlin_spca['p'], clamp_bias=nonlin_spca['c'], ln_k=ln_k
        ))
    elif no_ln: # If no_ln is selected, do not use layer norm
        rob_model = MappedModel(rob_model, IdentityMap()).to(device)
    else:       # Otherwise, use layer norm
        rob_model = MappedModel(rob_model, LayerNormMap(ln_k=ln_k)).to(device)
    std_model, rob_model = torch.nn.DataParallel(std_model), torch.nn.DataParallel(rob_model)

    epsilon = .5 if pgd_type == 'l_2' else 4. / 255. if dataset_name == 'imagenet' else 8. / 255.
    for model, mdl_name in zip([rob_model, std_model], ["ROB", "STD"]):

        acc, save_dic = margin_single_model(
            model=model, dataset_name=dataset_name, test_loader=test_loader, batch_size=batch_size,
            get_attacked_images_fn=get_attacked_images_fn, pgd_type=pgd_type,
            epsilon=epsilon, n_examples=n_examples, device=device, seed=SEED
        )
        save_mdl_name = rob_model_name if mdl_name == 'ROB' else 'std'
        appendix = '' if no_ln else '_map' if use_rob_map else '_ln'
        dir_name = f"{dataset_name}/{save_mdl_name}/{image_type}{appendix}"
        makedirs(f"results/{dir_name}", exist_ok=True)
        save_name = f"{dir_name}/{dataset_name}_{save_mdl_name}_{image_type}{appendix}"

        print(f"\nAccuracy of the {mdl_name} network on the "
              f"{n_examples} {image_type} test images: {acc:.3f} %.")
        with open(f"results/{save_name}_summary.csv", 'w') as summ_file:
            summ_file.write(
                f"Accuracy of the {mdl_name} network on the "
                f"{n_examples} {image_type} test images: {acc:.3f} %.\n"
            )

        save_margin(save_dic, save_name, n_examples)


@click.command(context_settings={'show_default': True})
@click.option(
    '--dataset_name', required=True, type=click.Choice(['cifar10', 'cifar100', 'imagenet']),
    help="The dataset to use. One of {'cifar10', 'cifar100', 'imagenet'}."
)
@click.option(
    '--pair/--single', is_flag=True, default=False,
    help="If True, calculate margins for the std model in addition to model_name."
)
@click.option(
    '--model_name', type=str, required=True,
    help="The base robust model to experiment with. Options include "
         "{std', 'Gowal2020Uncovering_extra', 'Wang2023Better_WRN-70-16', 'Peng2023Robust'}."
)
@click.option(
    '--use_rob_map/--no_rob_map', is_flag=True, default=False,
    help="If True, use the optimal nonlinear transformation for the robust base model."
)
@click.option(
    '--use_ln/--no_ln', is_flag=True, default=True,
    help="If True, use layer normalization for the base model logits."
)
@click.option(
    '--image_type', type=click.Choice(['clean', 'pgd20', 'aa']), required=True,
    help="Type of images to experiment with. One of {'clean', 'pgd20', 'aa'}."
)
@click.option(
    '--pgd_type', type=click.Choice(['l_inf', 'l_2']), default='l_inf', show_default=True,
    help="Type of images to experiment with. One of {'l_inf', 'l_2'}. Default to 'l_inf'."
)
@click.option(
    '--n_examples', type=int, default=10000, show_default=True, 
    help="Number of evaluation examples."
)
@click.option(
    '--batch_size', type=int, default=1000, show_default=True, help="Evaluation batch size."
)
@click.option(
    '--ln_k', type=int, default=250, show_default=True,
    help="The top-k option for layer norm. Default to 250."
)
@click.option(
    '--device', default=(
        'cuda' if torch.cuda.is_available() else 
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
)
def calc_margin(
    dataset_name, pair, model_name, use_rob_map, use_ln,
    image_type, pgd_type, n_examples, batch_size, ln_k, device
):
    print(f"Running on {device} device.")
    margin_fn = get_margin_pair if pair else get_margin_single
    margin_fn(
        dataset_name, model_name, use_rob_map, not use_ln,
        image_type, pgd_type, n_examples, batch_size, ln_k, device
    )


if __name__ == "__main__":
    seed_all(SEED)
    calc_margin()
