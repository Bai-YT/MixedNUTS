# Copyright (c) 2020-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from autoattack.other_utils import L0_norm, L1_norm, L2_norm
from autoattack.checks import check_zero_gradients
from autoattack.autopgd_base import L1_projection

from utils.misc_utils import seed_all


class AdaptiveAPGDAttack:
    """
    AutoPGD
    https://arxiv.org/abs/2003.01690

    :param predict:       forward pass function
    :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           bound on the norm of perturbations
    :param seed:          random seed for the starting point
    :param loss:          loss to optimize ('ce', 'dlr' supported)
    :param eot_iter:      iterations for Expectation over Trasformation
    :param rho:           parameter for decreasing the step size
    """
    def __init__(
        self,
        predict,
        n_iter=100,
        norm="Linf",
        n_restarts=1,
        eps=None,
        seed=0,
        loss="ce",
        eot_iter=1,
        rho=0.75,
        topk=None,
        verbose=False,
        device=None,
        use_largereps=False,
        is_tf_model=False,
        logger=None,
    ):
        """
        AutoPGD implementation in PyTorch
        """
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        self.use_largereps = use_largereps
        self.n_iter_orig = n_iter + 0
        self.eps_orig = float(eps)
        self.is_tf_model = is_tf_model
        self.y_target = None
        self.logger = logger

        assert self.norm in ["Linf", "L2", "L1"]
        assert not self.eps is None

        ### set parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)
        self.is_mixed_classifier = (
            hasattr(self.model, 'std_map') or (
                hasattr(self.model, 'module') and hasattr(self.model.module, 'std_map')
            )
        )

    def init_hyperparam(self, x):
        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def check_oscillation(self, x, j, k, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
            t += (x[j - counter5] > x[j - counter5 - 1]).float()
        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == "Linf":
            t = x.abs().view(x.shape[0], -1).max(1)[0]
        elif self.norm == "L2":
            t = (x**2).view(x.shape[0], -1).sum(-1).sqrt()
        elif self.norm == "L1":
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)

        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (
            x_sorted[:, -1] - x_sorted[:, -3] + 1e-12
        )

    def get_grad_and_logits(self, x, y, criterion_indiv):
        x.requires_grad_()
        grad = torch.zeros_like(x)

        for _ in range(self.eot_iter):
            if self.is_tf_model:  # Tensorflow
                logits, loss_indiv, grad_curr = (
                    criterion_indiv(x, y) if self.y_target is None
                    else criterion_indiv(x, y, self.y_target)
                )
                grad += grad_curr
            else:  # PyTorch
                with torch.enable_grad():
                    if self.is_mixed_classifier:
                        logits, logits_diffable, _ = self.model(x, return_all=True)
                    else:
                        logits = self.model(x)
                        logits_diffable = logits

                    loss_indiv = criterion_indiv(logits, y)
                    loss_indiv_diffable = criterion_indiv(logits_diffable, y)
                    loss_diffable = loss_indiv_diffable.sum()

                grad += torch.autograd.grad(loss_diffable, [x])[0].detach()

        grad /= float(self.eot_iter)
        return grad, loss_indiv, logits
    
    def attack_single_run(self, x, y, x_init=None):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        # Pre-specified initialization
        if x_init is not None:
            x_adv = x_init.clone()
            if self.norm == "L1" and self.verbose:
                info = (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()
                print(f"[custom init] L1 perturbation {info:.5f}")
        # Random initialization
        elif self.norm == "Linf":
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.eps * torch.ones_like(x).detach() * self.normalize(t)
        elif self.norm == "L2":
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x).detach() * self.normalize(t)
        elif self.norm == "L1":
            t = torch.randn(x.shape).to(self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        # Keep track of the minimum margin achieved throughout the optimization
        min_margin = torch.ones(x.shape[0], device=x.device)

        loss_steps = torch.zeros([self.n_iter, x.shape[0]]).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]).to(self.device)
        robust_mask_steps = torch.zeros_like(loss_best_steps)

        # Define loss function
        if self.is_tf_model:  # Tensorflow model
            if self.loss == "ce":
                criterion_indiv = self.model.get_logits_loss_grad_xent
            elif self.loss == "dlr":
                criterion_indiv = self.model.get_logits_loss_grad_dlr
            elif self.loss == "dlr-targeted":
                criterion_indiv = self.model.get_logits_loss_grad_target
            else:
                raise ValueError("unknown loss")
        else:  # PyTorch model
            if self.loss == "ce":
                criterion_indiv = nn.CrossEntropyLoss(reduction="none")
            elif self.loss == "ce-targeted-cfts":
                criterion_indiv = lambda x, y: \
                    -1. * F.cross_entropy(x, y, reduction="none")
            elif self.loss == "dlr":
                criterion_indiv = self.dlr_loss
            elif self.loss == "dlr-targeted":
                criterion_indiv = self.dlr_loss_targeted
            elif self.loss == "ce-targeted":
                criterion_indiv = self.ce_loss_targeted
            else:
                raise ValueError("unknown loss")

        ### get gradient
        grad, loss_indiv, logits = self.get_grad_and_logits(x_adv, y, criterion_indiv)

        grad_best = grad.clone()
        if self.loss in ["dlr", "dlr-targeted"]:
            # check if there are zero gradients
            check_zero_gradients(grad, logger=self.logger)

        # Initialize statistics
        robust_mask = logits.detach().argmax(1) == y
        robust_mask_steps[0] = robust_mask
        loss_best = loss_indiv.detach().clone()

        # Calculate step size
        alpha = 2. if self.norm in ["Linf", "L2"] else \
            1. if self.norm in ["L1"] else 2e-2
        step_size = alpha * self.eps * torch.ones([x.shape[0], *([1] * self.ndims)]) \
            .to(self.device).detach()
        x_adv_old = x_adv.clone()

        k = self.n_iter_2
        n_fts = math.prod(self.orig_dim)
        if self.norm == "L1":
            k = max(int(0.04 * self.n_iter), 1)
            if x_init is None:
                topk = 0.2 * torch.ones([x.shape[0]], device=self.device)
                sp_old = n_fts * torch.ones_like(topk)
            else:
                topk = L0_norm(x_adv - x) / n_fts / 1.5
                sp_old = L0_norm(x_adv - x)
            adasp_redstep = 1.5
            adasp_minstep = 10.
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

        u = torch.arange(x.shape[0], device=self.device)
        ### Perform the iterative APGD updates
        for i in range(self.n_iter):
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                a = 0.75 if i > 0 else 1.

                # Perform the optimization step
                if self.norm == "Linf":
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(
                        torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0., 1.
                    )
                    x_adv_1 = torch.clamp(
                        torch.min(torch.max(
                            x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - self.eps,
                        ), x + self.eps,), 0., 1.
                    )
                elif self.norm == "L2":
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    x_adv_1 = torch.clamp(
                        x + self.normalize(x_adv_1 - x) * torch.min(
                            self.eps * torch.ones_like(x).detach(),
                            L2_norm(x_adv_1 - x, keepdim=True),
                        ), 0., 1.
                    )
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(
                        x + self.normalize(x_adv_1 - x)
                        * torch.min(
                            self.eps * torch.ones_like(x).detach(),
                            L2_norm(x_adv_1 - x, keepdim=True),
                        ), 0., 1.
                    )
                elif self.norm == "L1":
                    grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                    topk_curr = torch.clamp(
                        (1. - topk) * n_fts, min=0, max=n_fts - 1
                    ).long()
                    grad_topk = grad_topk[u, topk_curr].view(
                        -1, *([1] * (len(x.shape) - 1))
                    )
                    sparsegrad = grad * (grad.abs() >= grad_topk).float()
                    x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                        L1_norm(sparsegrad.sign(), keepdim=True) + 1e-10
                    )
                    delta_u = x_adv_1 - x
                    delta_p = L1_projection(x, delta_u, self.eps)
                    x_adv_1 = x + delta_u + delta_p

                x_adv = x_adv_1

            ### get gradient for the next iteration
            grad, loss_indiv, logits = self.get_grad_and_logits(x_adv, y, criterion_indiv)

            ### Update attacked images based on margins
            with torch.no_grad():
                correct_mask = logits.detach().argmax(dim=1) == y  # Correct entries this iteration (mask)
                robust_mask = torch.min(robust_mask, correct_mask)  # Robust entries so far (mask)
                robust_mask_steps[i + 1] = robust_mask

                ind_incorr = (correct_mask == 0).nonzero().reshape(-1)  # Incorrect indices
                ind_robust = robust_mask.nonzero().reshape(-1)  # Robust so far indices

                # If an input is successfully perturbed, save it
                # x_best_adv[ind_incorr] = x_adv[ind_incorr]
                # If an input has not been successfully perturbed so far, then also save it
                # x_best_adv[ind_robust] = x_adv[ind_robust]  # This is new in this implementation.

                # Save min margin attacked images (new in this implementation)
                probs = logits.detach().softmax(dim=1)
                probs_incorr = probs[ind_incorr].reshape(-1, probs.shape[1])
                probs_robust = probs[ind_robust].reshape(-1, probs.shape[1])

                # For incorrect examples, the margin should be negative
                prob_incorr_gt = probs_incorr.gather(dim=1, index=y[ind_incorr].unsqueeze(1))
                margin_incorr = prob_incorr_gt.reshape(-1) - probs_incorr.max(dim=1)[0]
                ind_to_update_1 = (margin_incorr < min_margin[ind_incorr]).nonzero().reshape(-1)
                min_margin[ind_incorr[ind_to_update_1]] = margin_incorr[ind_to_update_1].float()
                x_best_adv[ind_incorr[ind_to_update_1]] = x_adv[ind_incorr[ind_to_update_1]].float()

                # For robust examples, the margin should be positive
                top2_robust = probs_robust.topk(k=2, dim=1).values
                margin_robust = top2_robust[:, 0] - top2_robust[:, 1]
                ind_to_update_2 = (margin_robust < min_margin[ind_robust]).nonzero().reshape(-1)
                min_margin[ind_robust[ind_to_update_2]] = margin_robust[ind_to_update_2].float()
                x_best_adv[ind_robust[ind_to_update_2]] = x_adv[ind_robust[ind_to_update_2]].float()

            if self.verbose:  # Print out optimization info
                str_stats = (
                    f" - step size: {step_size.mean():.5f} - topk: {(topk.mean() * n_fts):.2f}"
                    if self.norm in ["L1"] else ""
                )
                print(f"[m] iteration: {i} - best loss: {loss_best.sum():.6f} - "
                      f"robust accuracy: {robust_mask.float().mean():.2%}{str_stats}")

            ### check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1
                ind = (y1 > loss_best).nonzero().squeeze()

                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind]
                loss_best_steps[i + 1] = loss_best

                counter3 += 1

                if counter3 == k:
                    if self.norm in ["Linf", "L2"]:
                        fl_oscillation = self.check_oscillation(
                            x=loss_steps, j=i, k=k, k3=self.thr_decr
                        )
                        fl_reduce_no_impr = (1. - reduced_last_check) * (
                            loss_best_last_check >= loss_best
                        ).float()
                        fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
                        reduced_last_check = fl_oscillation.clone()
                        loss_best_last_check = loss_best.clone()

                        if fl_oscillation.sum() > 0:
                            ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                            step_size[ind_fl_osc] /= 2.
                            x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                            grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()
                        k = max(k - self.size_decr, self.n_iter_min)

                    elif self.norm == "L1":
                        sp_curr = L0_norm(x_best - x)
                        fl_redtopk = (sp_curr / sp_old) < 0.95
                        topk = sp_curr / n_fts / 1.5
                        step_size[fl_redtopk] = alpha * self.eps
                        step_size[~fl_redtopk] /= adasp_redstep
                        step_size.clamp_(
                            alpha * self.eps / adasp_minstep, alpha * self.eps
                        )
                        sp_old = sp_curr.clone()

                        x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                        grad[fl_redtopk] = grad_best[fl_redtopk].clone()

                    counter3 = 0

        return x_best, robust_mask, loss_best, x_best_adv, min_margin

    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
        assert len(epss) == len(iters)
        assert self.norm in ["L1"]

        self.use_rs = False
        if not use_rs:
            x_init = None
        else:
            x_init = x + torch.randn_like(x)
            x_init += L1_projection(x, x_init - x, float(epss[0]))

        if self.verbose:
            print(f"total iter: {sum(iters)}")

        for eps, niter in zip(epss, iters):
            if self.verbose:
                print(f"using eps: {eps:.2f}")
            self.n_iter = niter + 0
            self.eps = eps + 0.

            if not x_init is None:
                x_init += L1_projection(x, x_init - x, float(eps))

            x_init, acc, loss, x_adv, min_margin = \
                self.attack_single_run(x, y, x_init=x_init)

        return x_init, acc, loss, x_adv, min_margin

    def perturb(self, x, y=None, best_loss=False):
        """
        :param x:           Clean images.
        :param y:           Clean labels. If None, use the predicted labels.
        :param best_loss:   If True, return the points attaining highest loss.
                            Otherwise, return minimum margin adversarial examples.
        """
        assert self.loss in ["ce", "dlr"]
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        y_pred = (self.model.predict(x) if self.is_tf_model else self.model(x)).argmax(1)
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        if self.loss != "ce-targeted":
            acc = y_pred == y
        else:
            acc = y_pred != y

        # Initialize minimum margin
        min_margin = torch.ones(x.shape[0], device=x.device)

        if self.verbose:
            print("--------------------------",
                  f" running {self.norm}-attack with epsilon {self.eps:.5f} ",
                  "--------------------------")
            print(f"initial accuracy: {acc.float().mean():.2%}")

        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [0.3 * self.n_iter_orig, 0.3 * self.n_iter_orig, 0.4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            # make sure to use the given iterations
            iters[-1] = self.n_iter_orig - sum(iters[:-1])
            if self.verbose:
                print(f"using schedule [{'+'.join([str(c) for c in epss])}x"
                      f"{'+'.join([str(c) for c in iters])}]")

        startt = time.time()
        if not best_loss:
            seed_all(self.seed)

            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)

                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()

                    best_curr, acc_curr, loss_curr, adv_curr, cur_min_margin = (
                        self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                        if self.use_largereps else
                        self.attack_single_run(x_to_fool, y_to_fool)
                    )

                    # Update minimum margin and return minimum margin images.
                    # This is different from the original implementation.
                    ind_to_update = (cur_min_margin < min_margin[ind_to_fool]).nonzero().squeeze()
                    min_margin[ind_to_fool[ind_to_update]] = cur_min_margin[ind_to_update]
                    adv[ind_to_fool[ind_to_update]] = adv_curr[ind_to_update].detach().clone()

                    # Update the accuracy matrix
                    ind_curr = (acc_curr == 0).nonzero().squeeze()
                    acc[ind_to_fool[ind_curr]] = 0

                    if self.verbose:
                        print(f"restart {counter} - robust accuracy: {acc.float().mean():.2%}"
                              f" - cum. time: {(time.time() - startt):.1f} s")
            return adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float("inf"))

            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr]
                loss_best[ind_curr] = loss_curr[ind_curr]

                if self.verbose:
                    print(f"restart {counter} - loss: {loss_best.sum():.5f}"
                          f" - cum. time: {(time.time() - startt):.1f} s")
            return adv_best


class AdaptiveAPGDAttack_targeted(AdaptiveAPGDAttack):
    def __init__(
        self,
        predict,
        n_iter=100,
        norm="Linf",
        n_restarts=1,
        eps=None,
        seed=0,
        eot_iter=1,
        rho=0.75,
        topk=None,
        n_target_classes=9,
        verbose=False,
        device=None,
        use_largereps=False,
        is_tf_model=False,
        logger=None,
    ):
        """
        AutoPGD on the targeted DLR loss
        """
        super().__init__(
            predict,
            n_iter=n_iter,
            norm=norm,
            n_restarts=n_restarts,
            eps=eps,
            seed=seed,
            loss="dlr-targeted",
            eot_iter=eot_iter,
            rho=rho,
            topk=topk,
            verbose=verbose,
            device=device,
            use_largereps=use_largereps,
            is_tf_model=is_tf_model,
            logger=logger,
        )
        self.y_target = None
        self.n_target_classes = n_target_classes

    def dlr_loss_targeted(self, x, y):
        x_sorted, _ = x.sort(dim=1)
        denominator = x_sorted[:, -1] - (x_sorted[:, -3] + x_sorted[:, -4]) / 2 + 1e-12
        x_y = torch.gather(x, 1, y.unsqueeze(-1)).squeeze(-1)
        x_target = torch.gather(x, 1, self.y_target.unsqueeze(-1)).squeeze(-1)
        return (x_target - x_y) / denominator

    def ce_loss_targeted(self, x, y):
        return -1. * F.cross_entropy(x, self.y_target, reduction="none")

    def perturb(self, x, y=None):
        """
        :param x:   Clean images.
        :param y:   Clean labels. If None, use the predicted labels.
        """
        assert self.loss in ["dlr-targeted"]  # 'ce-targeted'
        if y is not None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        y_pred = (self.model.predict(x) if self.is_tf_model else self.model(x)).argmax(1)
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        acc = y_pred == y
        if self.verbose:
            print("--------------------------",
                  f" running {self.norm}-attack with epsilon {self.eps:.5f} ",
                  "--------------------------")
            print(f"initial accuracy: {acc.float().mean():.2%}")

        startt = time.time()
        seed_all(self.seed)

        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [0.3 * self.n_iter_orig, 0.3 * self.n_iter_orig, 0.4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1])
            if self.verbose:
                print(f"using schedule [{'+'.join([str(c) for c in epss])}x"
                      f"{'+'.join([str(c) for c in iters])}]")

        # Initialize minimum margin
        min_margin = torch.ones(x.shape[0], device=x.device)

        for target_class in range(2, self.n_target_classes + 2):
            for counter in range(self.n_restarts):

                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)

                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    output = (
                        self.model.predict(x_to_fool) if self.is_tf_model 
                        else self.model(x_to_fool)
                    )
                    self.y_target = output.sort(dim=1)[1][:, -target_class]

                    best_curr, acc_curr, loss_curr, adv_curr, cur_min_margin = (
                        self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                        if self.use_largereps else
                        self.attack_single_run(x_to_fool, y_to_fool)
                    )

                    # Update minimum margin and return minimum margin images.
                    # This is different from the original implementation.
                    ind_to_update = (cur_min_margin < min_margin[ind_to_fool]).nonzero().squeeze()
                    min_margin[ind_to_fool[ind_to_update]] = cur_min_margin[ind_to_update]
                    adv[ind_to_fool[ind_to_update]] = adv_curr[ind_to_update].detach().clone()

                    ind_curr = (acc_curr == 0).nonzero().squeeze()
                    acc[ind_to_fool[ind_curr]] = 0
                    if self.verbose:
                        print(f"target class {target_class}",
                              f" - restart {counter} - robust accuracy: {acc.float().mean():.2%}"
                              f" - cum. time: {(time.time() - startt):.1f} s")
        return adv
