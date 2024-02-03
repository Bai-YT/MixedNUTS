import click
import yaml
from tqdm import tqdm

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from models.output_maps import LNClampPowerScaleMap, LNPowerScaleMap, ScaleMap, LayerNormMap
from utils.model_utils import load_rob_model
from utils.spca_utils import visualize_obj, get_margin_file
from utils.misc_utils import seed_all

SEED = 20230331


@click.command(context_settings={'show_default': True})
@click.option(
    '--dataset_name', required=True,
    type=click.Choice(['cifar10', 'cifar100', 'imagenet']),
    help="The dataset to use. One of {'cifar10', 'cifar100', 'imagenet'}."
)
@click.option(
    '--rob_model_name', type=str, required=True,
    help="The robust base model to use. Options include "
         "{'Gowal2020Uncovering_extra', 'Wang2023Better_WRN-70-16', 'Peng2023Robust'}."
)
@click.option(
    '--cln_file_path', type=str, default=None,
    help="Location of the saved clean margin files. "
         "If None, use default location (see get_margin_file). Default to None."
)
@click.option(
    '--atk_file_path', type=str, default=None,
    help="Location of the saved attacked margin files. "
         "If None, use default location (see get_margin_file). Default to None."
)
@click.option(
    '--target_rob_beta', type=float, required=True, show_default=True,
    help="Target robust accuracy of the mixed classifier in percentage."
)
@click.option(
    '--nonlin_type', required=True,
    type=click.Choice(['gelu', 'relu', 'elu', 'softplus', 'linear', 'ts', 'none', 'ln']),
    help="The nonlinearity used to build the transformed robust model. "
         "One of {'gelu', 'relu', 'elu', 'softplus', 'ln', 'ts', 'none'}."
         "If 'ts', no clamping is used; temperature scaling is allowed; layer norm is disabled."
         "If 'ln', no clamping is used; temperature scaling is disabled; layer norm is applied."
         "If 'none', no clamping is used; temperature scaling is disabled; layer norm is disabled."
)
@click.option(
    '--ln_k', type=int, default=250, show_default=True,
    help="The top-k option for layer norm. Default to 250."
)
@click.option(
    '--logits_dir', type=str, default=None,
    help="Location of the saved logits. If None, calculate the logits from the model."
         "Default to None."
)
@click.option(
    '--batch_size', default=1000, show_default=True, help="Evaluation batch size."
)
@click.option(
    '--device', default=(
        'cuda' if torch.cuda.is_available() else 
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
)

def calc_spca(
    dataset_name, rob_model_name, cln_file_path, atk_file_path,
    target_rob_beta, nonlin_type, ln_k, logits_dir, batch_size, device
):
    # Load the robust model
    orig_rob_model = load_rob_model(
        rob_model_name=rob_model_name, dataset_name=dataset_name,
        parallelize=True, device=device
    )
    clamp_fns = {'gelu': nn.GELU, 'relu': nn.ReLU, 'elu': nn.ELU, 'softplus': nn.Softplus}

    # Calculate the logits of the raw model
    if logits_dir is not None:
        logits_dic = torch.load(f"results/{dataset_name}/{rob_model_name}/logits.pt")
        atk_corr_logits, cln_incorr_logits = logits_dic['atk_corr'], logits_dic['cln_incorr']

    else:
        # Load clean margin file
        cln_dic = get_margin_file(
            custom_file_path=cln_file_path, dataset_name=dataset_name,
            rob_model_name=rob_model_name, image_type='clean_ln'
        )
        # Split the incorrectly predicted clean images into batches
        cln_incorr_imgs = torch.tensor(cln_dic["images"]["incorrect"])
        cln_incorr_labs = torch.tensor(cln_dic["labels"]["incorrect"])
        assert cln_incorr_imgs.shape[0] > 0, "No incorrectly predicted clean images."
        num_cln_incorr_batches = math.ceil(len(cln_incorr_imgs) / batch_size)
        cln_incorr_imgs = cln_incorr_imgs.chunk(num_cln_incorr_batches, dim=0)

        # Load AutoAttacked margin file
        atk_dic = get_margin_file(
            custom_file_path=atk_file_path, dataset_name=dataset_name,
            rob_model_name=rob_model_name, image_type='aa_ln'
        )
        # Split the correctly predicted AutoAttacked images into batches
        atk_corr_imgs = torch.tensor(atk_dic["images"]["correct"])
        atk_corr_labs = torch.tensor(atk_dic["labels"]["correct"])
        assert atk_corr_imgs.shape[0] > 0, "No correctly predicted attacked images."
        num_atk_corr_batches = math.ceil(len(atk_corr_imgs) / batch_size)
        atk_corr_imgs = atk_corr_imgs.chunk(num_atk_corr_batches, dim=0)

        atk_corr_logits, cln_incorr_logits = [], []
        for atk_imgs in tqdm(atk_corr_imgs):
            with torch.no_grad():
                atk_corr_logits += [orig_rob_model(atk_imgs.to(device)).detach()]
        for cln_imgs in tqdm(cln_incorr_imgs):
            with torch.no_grad():
                cln_incorr_logits += [orig_rob_model(cln_imgs.to(device)).detach()]
        atk_corr_logits = torch.cat(atk_corr_logits, dim=0)
        cln_incorr_logits = torch.cat(cln_incorr_logits, dim=0)
        logits_dic = {
            'atk_corr': atk_corr_logits.cpu(), 'cln_incorr': cln_incorr_logits.cpu()
        }
        torch.save(logits_dic, f"results/{dataset_name}/{rob_model_name}/logits.pt")

    # Initial guess for the grid
    with open('configs/spca_initial_guess.yaml', 'r') as file:
        init_dic = yaml.safe_load(file)[dataset_name][rob_model_name][nonlin_type]

    # Meshgrid for the grid search
    num_vars = 8
    # Scale
    if nonlin_type in ['none', 'ln']:
        ss = np.array([1])
    else:
        ss = np.logspace(
            np.log10(init_dic['s_low']), np.log10(init_dic['s_high']), num_vars
        )
    # Power
    if nonlin_type in ['ts', 'none', 'ln']:
        pp = np.array([1])
    else:
        pp = np.linspace(init_dic['p_low'], init_dic['p_high'], num_vars)
    # Bias
    if nonlin_type in ['linear', 'ts', 'none', 'ln']:
        cc = np.array([0])
    else:
        cc = np.linspace(init_dic['c_low'], init_dic['c_high'], num_vars)

    # Initialize grids to store the resulting objective functions
    cutoffs = np.empty((len(ss), len(cc), len(pp)))
    obj_percents = np.empty((len(ss), len(cc), len(pp)))
    atk_corr_margins = np.empty((len(ss), len(cc), len(pp), atk_corr_labs.shape[0]))
    cln_incorr_margins = np.empty((len(ss), len(cc), len(pp), cln_incorr_labs.shape[0]))
    atk_corr_mean_margins = np.empty((len(ss), len(cc), len(pp)))
    cln_incorr_mean_margins = np.empty((len(ss), len(cc), len(pp)))
    atk_corr_median_margins = np.empty((len(ss), len(cc), len(pp)))
    cln_incorr_median_margins = np.empty((len(ss), len(cc), len(pp)))

    map_device = device if 'cuda' in device else 'cpu'
    atk_corr_logits = atk_corr_logits.to(map_device)
    cln_incorr_logits = cln_incorr_logits.to(map_device)

    for s_id, s in enumerate(tqdm(ss)):
        for c_id, c in enumerate(cc):
            for p_id, p in enumerate(pp):

                # Create non-linear mapping
                if nonlin_type in ['gelu', 'relu', 'elu', 'softplus']:
                    nonlin_map = LNClampPowerScaleMap(
                        scale=s, power=p, clamp_bias=c, clamp_fn=clamp_fns[nonlin_type](), ln_k=ln_k
                    )
                elif nonlin_type == 'linear':
                    nonlin_map = LNPowerScaleMap(scale=s, power=p, ln_k=ln_k)
                elif nonlin_type == 'ln':
                    nonlin_map = LayerNormMap(ln_k=ln_k)
                elif nonlin_type in ['none', 'ts']:
                    nonlin_map = ScaleMap(scale=s)
                else:
                    raise NotImplementedError

                # Calculate margins for correctly predicted attacked images
                atk_corr_probs = nonlin_map(atk_corr_logits, return_probs=True)
                assert not atk_corr_probs.isnan().any(), print(atk_corr_probs)
                atk_corr_top_2 = atk_corr_probs.topk(k=2, dim=1)
                atk_corr_predicted = atk_corr_top_2.indices[:, 0]
                cur_atk_corr_margins = atk_corr_top_2.values[:, 0] - atk_corr_top_2.values[:, 1]
                atk_corr_margins[s_id, c_id, p_id] = cur_atk_corr_margins.cpu().numpy()
                atk_corr_mean_margins[s_id, c_id, p_id] = cur_atk_corr_margins.mean().item()
                atk_corr_median_margins[s_id, c_id, p_id] = cur_atk_corr_margins.median().item()

                # Verify whether all attacked images are correctly predicted as expected
                num_incorr_atk = atk_corr_labs.size(0) - \
                    (atk_corr_predicted.cpu() == atk_corr_labs).sum()
                if num_incorr_atk != 0:
                    print(f"{num_incorr_atk.item()} attacked example unexpectedly "
                          f"incorrect for s={s}, c={c}, p={p}.")

                # Calculate margins for incorrectly predicted benign images
                cln_incorr_probs = nonlin_map(cln_incorr_logits, return_probs=True)
                assert not cln_incorr_probs.isnan().any(), print(cln_incorr_probs)
                cln_incorr_top_2 = cln_incorr_probs.topk(k=2, dim=1)
                cln_incorr_predicted = cln_incorr_top_2.indices[:, 0]
                cur_cln_incorr_margins = cln_incorr_top_2.values[:, 0] - cln_incorr_top_2.values[:, 1]
                cln_incorr_margins[s_id, c_id, p_id] = cur_cln_incorr_margins.cpu().numpy()
                cln_incorr_mean_margins[s_id, c_id, p_id] = cur_cln_incorr_margins.mean().item()
                cln_incorr_median_margins[s_id, c_id, p_id] = cur_cln_incorr_margins.median().item()
                    
                # Verify whether all clean images are incorrectly predicted as expected
                num_corr_cln = (cln_incorr_predicted.cpu() == cln_incorr_labs).sum()
                if num_corr_cln != 0:
                    print(f"{num_corr_cln.item()} clean examples unexpectedly "
                          f"correct for s={s}, c={c}, p={p}.")

                # Calculate the cutoff based on desired attacked accuracy
                # Cutoff is (1-\alpha)/\alpha in our paper.
                cur_cutoff = np.percentile(cur_atk_corr_margins.cpu().numpy(), 100. - target_rob_beta)
                cutoffs[s_id, c_id, p_id] = cur_cutoff

                # If the cutoff is close to 1, then the attacked margins are very large
                # and the objective function may be numerically unstable.
                # Worst case is that all margins are 1 and the objective becomes 0.
                # Therefore, we warn the user if the cutoff is close to 1 and 
                # subtract a small number from it for objective calculation.
                if cur_cutoff >= (1. - (1e-7 if dataset_name == 'imagenet' else 3e-6)):
                    print(f"Potential numerical issues for s={s}, c={c}, p={p}.")
                cur_cutoff = np.minimum(cur_cutoff, 1. - 1e-9)

                # Calculate the objective value of the optimization problem
                curr_obj = (cur_cln_incorr_margins >= cur_cutoff).double().mean().item()
                obj_percents[s_id, c_id, p_id] = curr_obj * 100

    # Save a video that visualizes the objective function
    save_name = f"{dataset_name}/{rob_model_name}/{nonlin_type}"
    # print(obj_percents[4, 1, -2])  # Estimate approx error for CIFAR-100
    visualize_obj(obj_percents, ss, cc, pp, save_name)

    # Find all combinations that optimizes the objective function
    # We want to use a numerically stable combination (1-alpha should not be tiny)
    idx0, idx1, idx2 = np.where(obj_percents == np.min(obj_percents))

    # Display the optimal combinations
    plt.figure(figsize=(8, 6))
    for idx in zip(idx0, idx1, idx2):
        s_star, c_star, p_star = ss[idx[0]], cc[idx[1]], pp[idx[2]]
        alpha_star = 1 / (1 + cutoffs[idx])
        print(f"Optimal s: {s_star}; c: {c_star}; p: {p_star}; "
              f"alpha: {alpha_star}; 1-alpha: {1 - alpha_star}.")
        print(f"Optimal objective (lower is better): {obj_percents[idx]} %.")
        print(f"{target_rob_beta}-percentile cutoff at optimal setting: {cutoffs[idx]}.\n")

        print("Mean / median margin for correct attacked predictions: "
              f"{atk_corr_mean_margins[idx]:.3f} / {atk_corr_median_margins[idx]:.3f}.")
        print("Mean / median margin for incorrect clean predictions: "
              f"{cln_incorr_mean_margins[idx]:.3f} / {cln_incorr_median_margins[idx]:.3f}.\n")
        plt.hist(
            atk_corr_margins[idx], bins=50,
            label=f'atk_corr_s={s_star:.1f}_c={c_star:.1f}_p={p_star:.1f}'
        )
        plt.hist(
            cln_incorr_margins[idx], bins=50,
            label=f'cln_incorr_s={s_star:.1f}_c={c_star:.1f}_p={p_star:.1f}'
        )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{save_name}_hist.pdf")


if __name__ == "__main__":
    seed_all(SEED)
    calc_spca()
