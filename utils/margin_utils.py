from tqdm import tqdm
from copy import deepcopy
import json, pickle
import os
from itertools import islice

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from adaptive_autoattack import adaptive_autoattack
from utils.pgd_attack import pgd_attack
from utils.data_utils import \
    CIFAR10_float, CIFAR100_float, ImageNet_float, ImageNetValSubset
from utils.model_utils import load_std_model, load_rob_model


def get_pgd20_attacked_images(
    model, dataset_name, test_loader, pgd_type, epsilon, n_examples, seed, device
):
    assert pgd_type in ['l_2', 'l_inf'], f"Unknown PGD type: {pgd_type}."
    num_batches = n_examples // test_loader.batch_size
    attacked_images, targets = [], []

    for im, tg in tqdm(islice(test_loader, num_batches), total=num_batches):
        im, tg = im.to(device), tg.to(device)

        pgd_loss = torch.nn.CrossEntropyLoss()
        att_im = pgd_attack(
            model, im, tg, pgd_type=pgd_type, pgd_eps=epsilon, pgd_iters=20,
            pgd_alpha=epsilon*0.086, pgd_loss=pgd_loss, device=device
        )
        attacked_images += [att_im.detach().cpu()]
        targets += [tg.detach().cpu()]

    return torch.cat(attacked_images, dim=0), torch.cat(targets, dim=0)


def get_autoattacked_images(
    model, dataset_name, test_loader, pgd_type, epsilon, n_examples, seed, device
):
    def set_aa_params():
        aa_adversary.apgd.n_restarts = 1
        aa_adversary.apgd_targeted.n_target_classes = 9
        aa_adversary.apgd_targeted.n_restarts = 1
        aa_adversary.fab.n_restarts = 1
        aa_adversary.fab.n_target_classes = 9
        aa_adversary.square.n_queries = 5000

    # Get attacked images
    if dataset_name == 'imagenet':
        images, targets = [], []
        num_batches = n_examples // test_loader.batch_size
        for im, tg in islice(test_loader, num_batches):
            images += [im]
            targets += [tg]
        images = torch.cat(images, dim=0).float()
        targets = torch.cat(targets, dim=0).long()
    else:
        images = test_loader.dataset.data[:n_examples, :, :, :]
        images = torch.from_numpy(images).float().permute(0, 3, 1, 2)
        targets = test_loader.dataset.targets[:n_examples]
        targets = torch.tensor(targets).long()

    # For margin calculation, we only use the two APGD modules of AutoAttack
    # since FAB-T and SQUARE do not successfully attack additional images.
    # attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    attacks_to_run = ['apgd-ce', 'apgd-t']
    assert pgd_type in ['l_2', 'l_inf'], f"Unknown PGD type: {pgd_type}."
    norm = {'l_2': 'L2', 'l_inf': 'Linf'}[pgd_type]
    aa_adversary = adaptive_autoattack.AdaptiveAutoAttack(
        model, norm=norm, eps=epsilon, version='custom',
        seed=seed, device=device, attacks_to_run=attacks_to_run
    )
    set_aa_params()

    attacked_images = aa_adversary.run_standard_evaluation(
        images, targets, bs=test_loader.batch_size
    ).detach().cpu()
    return attacked_images, targets


def get_adversarial_loader(
    dataset_name, test_loader, get_attacked_images_fn,
    model, pgd_type, epsilon, n_examples, seed, device
):
    """ Make a dataloader that loads attacked images. """

    adv_test_loader = deepcopy(test_loader)

    if get_attacked_images_fn is not None:
        # Query the attacks to get adversarial images
        print(f"Getting attacked images using {get_attacked_images_fn.__name__} "
              f"with {pgd_type} epsilon={epsilon}...")
        attacked_images, targets = get_attacked_images_fn(
            model=model, dataset_name=dataset_name,
            test_loader=test_loader, pgd_type=pgd_type, epsilon=epsilon,
            n_examples=n_examples, seed=seed, device=device
        )
        # Make adversarial dataloader
        if dataset_name == 'imagenet':
            adv_test_set = ImageNet_float(attacked_images, targets)
            adv_test_loader = torch.utils.data.DataLoader(
                adv_test_set, batch_size=test_loader.batch_size, shuffle=False
            )
        else:
            adv_test_loader.dataset.data[:n_examples] = \
                attacked_images.permute(0, 2, 3, 1).numpy()

    return adv_test_loader


def get_models_and_loader(
    dataset_name, rob_model_name, model_par_dir, dataset_dir,
    pgd_type='l_inf', batch_size=100, n_examples=10000, device=torch.device('cpu')
):
    """ Load the base classifiers for margin calculations. """

    if dataset_name == 'cifar10':
        # Build dataset
        num_classes = 10
        test_set = CIFAR10_float(
            root=dataset_dir, train=False, download=True, transform=transforms.ToTensor()
        )
        # Standard base model architecture and path
        std_model_arch = 'rn152'
        std_load_path = f"cifar10/cifar10_std_{std_model_arch}.pt"

    elif dataset_name == 'cifar100':
        # Build dataset
        num_classes = 100
        test_set = CIFAR100_float(
            root=dataset_dir, train=False, download=True, transform=transforms.ToTensor()
        )
        # Standard base model architecture and path
        std_model_arch = 'rn152'
        std_load_path = f"cifar100/cifar100_std_{std_model_arch}.pt"

    elif dataset_name == "imagenet":
        num_classes = 1000
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor()
        ])
        test_set = ImageNetValSubset(
            root=dataset_dir, transform=transform,
            images_per_class=int(np.ceil(n_examples / 1000))
        )
        std_model_arch = 'convnext_v2-l_224'
        std_load_path = "imagenet/imagenet_std_convnext_v2-l_224.pt"

    else:
        raise ValueError(f"{dataset_name} is not a supported dataset.")

    # Load standard base model
    std_load_path = os.path.join(model_par_dir, std_load_path)
    std_model = load_std_model(
        std_model_arch=std_model_arch, std_load_path=std_load_path, device=device,
        num_classes=num_classes, return_features=False, parallelize=True
    )
    # Load robust base model
    rob_model = load_rob_model(
        rob_model_name=rob_model_name, dataset_name=dataset_name,
        pgd_type=pgd_type, parallelize=True, device=device
    )

    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=(os.cpu_count() - 1 if dataset_name == "imagenet" else 0)
    )
    return std_model, rob_model, test_loader


def margin_single_model(
    model, dataset_name, test_loader, get_attacked_images_fn, pgd_type,
    epsilon, batch_size, n_examples, device=torch.device('cpu'), seed=0
):
    assert n_examples % batch_size == 0, \
        "Only support equal-sized batches. I.e., n_examples % batch_size must be 0."
    num_batches = n_examples // batch_size
    adv_test_loader = get_adversarial_loader(
        dataset_name, test_loader, get_attacked_images_fn,
        model, pgd_type, epsilon, n_examples, seed, device
    )

    # Initialize statistics
    correct, total, corr_margins, incorr_margins = 0, 0, [], []
    corr_images, corr_labels, corr_probs = [], [], []
    incorr_images, incorr_labels, incorr_probs = [], [], []

    # Calculate margins
    with torch.no_grad():
        for images, labels in tqdm(
            islice(adv_test_loader, num_batches), total=num_batches
        ):
            images, labels = images.to(device), labels.to(device)
            probs = model(images, return_probs=True).detach()
            assert not probs.isnan().any()
            top_2 = probs.topk(k=2, dim=1)
            predicted = top_2.indices[:, 0]

            corr_mask = predicted == labels.to(predicted.device)
            correct += corr_mask.sum().item()
            total += labels.size(0)

            corr_images += [images[corr_mask].cpu()]
            corr_labels += [labels[corr_mask].cpu()]
            corr_probs += [probs[corr_mask].cpu()]
            incorr_images += [images[~corr_mask].cpu()]
            incorr_labels += [labels[~corr_mask].cpu()]
            incorr_probs += [probs[~corr_mask].cpu()]

            margins = top_2.values[:, 0] - top_2.values[:, 1]
            corr_margins += [margins[corr_mask].cpu()]
            incorr_margins += [margins[~corr_mask].cpu()]

    assert total == n_examples, f"total={total} != n_examples={n_examples}."
    acc = 100 * correct / total

    corr_margins = torch.cat(corr_margins).numpy()
    corr_images = torch.cat(corr_images, dim=0).numpy()
    corr_labels = torch.cat(corr_labels).numpy()
    corr_probs = torch.cat(corr_probs, dim=0).numpy()
    incorr_margins = torch.cat(incorr_margins).numpy()
    incorr_images = torch.cat(incorr_images, dim=0).numpy()
    incorr_labels = torch.cat(incorr_labels).numpy()
    incorr_probs = torch.cat(incorr_probs, dim=0).numpy()

    return acc, {
        "margins": {"correct": corr_margins, "incorrect": incorr_margins},
        "images": {"correct": corr_images, "incorrect": incorr_images},
        "labels": {"correct": corr_labels, "incorrect": incorr_labels},
        "probs": {"correct": corr_probs, "incorrect": incorr_probs}
    }


def save_margin(save_dic, save_name, n_examples, save_all=True):
    # Save the margin array
    if save_all:
        print(f"Saving {'clean' if 'clean' in save_name else 'attacked'} images.")
        with open(f"results/{save_name}_{n_examples}examples.pt", 'wb') as pkl_file:
            pickle.dump(save_dic, pkl_file)
    else:
        print(f"{'Clean' if 'clean' in save_name else 'Attacked'} images not saved.")

    save_margins = save_dic['margins']
    save_margins['correct'] = list(save_margins["correct"].astype(float))
    save_margins['incorrect'] = list(save_margins["incorrect"].astype(float))
    with open(f"results/{save_name}_margins.json", 'w') as json_file:
        json.dump(save_margins, json_file, indent=4)

    # Print the mean and median
    corr_margins = save_margins["correct"]
    incorr_margins = save_margins["incorrect"]
    corr_mean_margins = np.mean(corr_margins)
    corr_median_margins = np.median(corr_margins)
    incorr_mean_margins = np.mean(incorr_margins)
    incorr_median_margins = np.median(incorr_margins)
    print("Mean / median margin for correct predictions: "
          f"{corr_mean_margins:.3f} / {corr_median_margins:.3f}.")
    print("Mean / median margin for incorrect predictions: "
          f"{incorr_mean_margins:.3f} / {incorr_median_margins:.3f}.\n")
    # Write to file
    with open(f"results/{save_name}_summary.csv", 'a') as summ_file:
        summ_file.write(
            "Mean / median margin for correct predictions: "
            f"{corr_mean_margins:.3f} / {corr_median_margins:.3f}.\n"
            "Mean / median margin for incorrect predictions: "
            f"{incorr_mean_margins:.3f} / {incorr_median_margins:.3f}.\n"
        )

    # Plot histograms
    plt.figure(figsize=(4, 3.5))
    bins = np.linspace(0, 1, 26)
    plt.hist(corr_margins, bins=bins, label="Correct", alpha=.8)
    plt.hist(incorr_margins, bins=bins, label="Incorrect", alpha=.6)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"results/{save_name}_{n_examples}examples.pdf", bbox_inches='tight')
