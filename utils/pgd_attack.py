import torch


def pgd_update(images, ori_images, pgd_type, pgd_alpha, pgd_eps, n, momentum=0):
    grad = images.grad + momentum

    if pgd_type == 'l_inf':
        adv_images = images.detach() + pgd_alpha * grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-pgd_eps, max=pgd_eps)  # Projection
    elif pgd_type == 'l_2':
        gradnorms = grad.reshape(n, -1).norm(dim=1, p=2).view(n, 1, 1, 1)
        adv_images = images.detach() + pgd_alpha * grad / gradnorms
        eta = adv_images - ori_images
        etanorms = eta.reshape(n, -1).norm(dim=1, p=2).view(n, 1, 1, 1)
        eta = eta * torch.minimum(torch.tensor(1.), pgd_eps / etanorms)  # Projection

    images = torch.clamp(ori_images + eta, min=0, max=1).detach()
    return images, grad


def pgd_attack(
    model, images, labels, pgd_type, pgd_loss, pgd_eps, pgd_alpha, pgd_iters,
    mom_decay=0, random_start=False, device=None
):
    assert pgd_type in ['l_inf', 'l_2']
    if device is None:
        device = images.device
    model.eval()

    ori_images = images.clone().detach()
    n, c, h, w = tuple(images.shape)
    if random_start:
        unit_noise = torch.empty_like(images).uniform_(-1, 1)
        images = images + unit_noise * pgd_eps
        images = torch.clamp(images, min=0, max=1).detach()

    momentum = 0  # Initial gradient momentum
    for _ in range(pgd_iters):
        model.zero_grad()
        images = images.clone().detach().requires_grad_(True)

        if hasattr(model, 'std_model'):  # model is a mixed classifier
            _, logits_diffable, _ = model(images, return_all=True)
        else:  # model is a conventional standalone classifier
            logits_diffable = model(images)

        if 'mps' in str(device):
            logits_diffable = logits_diffable.float().to(device)
        loss = pgd_loss(logits_diffable, labels).to(device)
        loss.backward()
        images, momentum = pgd_update(
            images, ori_images, pgd_type, pgd_alpha, pgd_eps, n, momentum * mom_decay
        )

    return images
