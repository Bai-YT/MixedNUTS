# Copyright (c) 2019-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

from autoattack.other_utils import zero_gradients
from autoattack.fab_base import FABAttack


class AdaptiveFABAttack_PT(FABAttack):
    """
    Fast Adaptive Boundary Attack (Linf, L2, L1)
    https://arxiv.org/abs/1907.02044
    
    :param predict:       forward pass function
    :param norm:          Lp-norm to minimize ('Linf', 'L2', 'L1' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           epsilon for the random restarts
    :param alpha_max:     alpha_max
    :param eta:           overshooting
    :param beta:          backward step
    """
    def __init__(
            self,
            model,
            norm='Linf',
            n_restarts=1,
            n_iter=100,
            eps=None,
            alpha_max=0.1,
            eta=1.05,
            beta=0.9,
            loss_fn=None,
            verbose=False,
            seed=0,
            targeted=False,
            device=None,
            n_target_classes=9):
        """ FAB-attack implementation in pytorch """

        self.model = model
        super().__init__(norm,
                         n_restarts,
                         n_iter,
                         eps,
                         alpha_max,
                         eta,
                         beta,
                         loss_fn,
                         verbose,
                         seed,
                         targeted,
                         device,
                         n_target_classes)

        self.is_mixed_classifier = (
            hasattr(self.model, 'std_map') or 
            (hasattr(self.model, 'module') and hasattr(self.model.module, 'std_map'))
        )

    def _predict_fn(self, x):
        return self.model(x)

    def _get_predicted_label(self, x):
        with torch.no_grad():
            outputs = self._predict_fn(x)
        _, y = torch.max(outputs, dim=1)
        return y

    def get_diff_logits_grads_batch(self, imgs, la):
        im = imgs.clone().requires_grad_()
        with torch.enable_grad():
            if self.is_mixed_classifier:
                logits, logits_diffable, _ = self.model(im, return_all=True)
            else:
                logits = self.model(im)
                logits_diffable = logits

        g2 = torch.zeros([y.shape[-1], *imgs.size()]).to(self.device)
        grad_mask = torch.zeros_like(logits)
        for counter in range(logits.shape[-1]):
            zero_gradients(im)
            grad_mask[:, counter] = 1.0
            logits_diffable.backward(grad_mask, retain_graph=True)
            grad_mask[:, counter] = 0.0
            g2[counter] = im.grad.detach().clone()

        g2 = torch.transpose(g2, 0, 1).detach()
        y2 = logits.detach()
        df = y2 - y2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
        dg = g2 - g2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
        df[torch.arange(imgs.shape[0]), la] = 1e10
        return df, dg

    def get_diff_logits_grads_batch_targeted(self, imgs, la, la_target):
        u = torch.arange(imgs.shape[0])
        im = imgs.clone().requires_grad_()

        with torch.enable_grad():
            if self.is_mixed_classifier:
                logits, logits_diffable, _ = self.model(im, return_all=True)
            else:
                logits = self.model(im)
                logits_diffable = logits

            diffy = -(logits[u, la] - logits[u, la_target])
            diffy_diffable = -(logits_diffable[u, la] - logits_diffable[u, la_target])
            sumdiffy_diffable = diffy_diffable.sum()

        zero_gradients(im)
        sumdiffy_diffable.backward()
        graddiffy = im.grad.detach().clone()
        df = diffy.detach().unsqueeze(1)
        dg = graddiffy.unsqueeze(1)
        return df, dg
