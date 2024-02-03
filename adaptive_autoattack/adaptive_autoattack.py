import time
import numpy as np

import torch
import torch.nn.functional as F

from autoattack import AutoAttack, checks
from autoattack.state import EvaluationState
from autoattack.other_utils import Logger


class AdaptiveAutoAttack(AutoAttack):
    def __init__(
        self, model, norm='Linf', eps=.3, seed=None, verbose=True, attacks_to_run=[],
        version='standard', is_tf_model=False, device=torch.device('cpu'), log_path=None
    ):
        self.model = model
        self.norm = norm
        assert norm in ['Linf', 'L2', 'L1']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.version = version
        self.is_tf_model = is_tf_model
        self.device = device
        self.logger = Logger(log_path)

        print("####### Starting Adaptive AutoAttack #######")
        if version in ['standard', 'plus', 'rand'] and attacks_to_run != []:
            raise ValueError("attacks_to_run will be overridden unless you use version='custom'")
        assert not self.is_tf_model, "Adaptive AutoAttack only supports PyTorch models for now."

        from adaptive_autoattack.autopgd_base import AdaptiveAPGDAttack
        self.apgd = AdaptiveAPGDAttack(
            self.model, n_restarts=5, n_iter=100, verbose=False,
            eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed,
            device=self.device, logger=self.logger
        )
        from .fab_pt import AdaptiveFABAttack_PT
        self.fab = AdaptiveFABAttack_PT(
            self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
            norm=self.norm, verbose=False, device=self.device
        )
        from autoattack.square import SquareAttack
        self.square = SquareAttack(
            self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
            n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False
        )
        from .autopgd_base import AdaptiveAPGDAttack_targeted
        self.apgd_targeted = AdaptiveAPGDAttack_targeted(
            self.model, n_restarts=1, n_iter=100, verbose=False, eps=self.epsilon, norm=self.norm, 
            eot_iter=1, rho=.75, seed=self.seed, device=self.device, logger=self.logger
        )

        if version in ['standard', 'plus', 'rand']:
            self.set_version(version)

    def run_standard_evaluation(
        self, x_orig, y_orig, bs=250, return_labels=False, state_path=None
    ):
        if state_path is not None and state_path.exists():
            state = EvaluationState.from_disk(state_path)
            if set(self.attacks_to_run) != state.attacks_to_run:
                raise ValueError("The state was created with a different set of attacks "
                                 "to run. You are probably using the wrong state file.")
            if self.verbose:
                self.logger.log(f"Restored state from {state_path}")
                self.logger.log(
                    "Since the state has been restored, **only** the adversarial "
                    "examples from the current run are going to be returned."
                )
        else:
            state = EvaluationState(set(self.attacks_to_run), path=state_path)
            state.to_disk()
            if self.verbose and state_path is not None:
                self.logger.log(f"Created state in {state_path}")                                

        attacks_to_run = list(
            filter(lambda attack: attack not in state.run_attacks, self.attacks_to_run)
        )
        if self.verbose:
            self.logger.log(f"using {self.version} version including {', '.join(attacks_to_run)}.")
            if state.run_attacks:
                self.logger.log(f"{', '.join(state.run_attacks)} was/were already run.")

        # checks on type of defense
        if self.version != 'rand':
            checks.check_randomized(self.get_logits, x_orig[:bs].to(self.device),
                y_orig[:bs].to(self.device), bs=bs, logger=self.logger)

        n_cls = checks.check_range_output(
            self.get_logits, x_orig[:bs].to(self.device), logger=self.logger
        )
        checks.check_dynamic(
            self.model, x_orig[:bs].to(self.device), self.is_tf_model, logger=self.logger
        )
        checks.check_n_classes(
            n_cls, self.attacks_to_run, self.apgd_targeted.n_target_classes,
            self.fab.n_target_classes, logger=self.logger
        )

        with torch.no_grad():
            # calculate initial accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))

            if state.robust_flags is None:
                robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
                y_adv = torch.empty_like(y_orig)

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])

                    x_batch = x_orig[start_idx:end_idx, :].clone().to(self.device)
                    y_batch = y_orig[start_idx:end_idx].clone().to(self.device)
                    pred_batch = self.get_logits(x_batch).argmax(dim=1)
                    y_adv[start_idx: end_idx] = pred_batch
                    correct_batch = (pred_batch == y_batch).detach().to(robust_flags.device)
                    robust_flags[start_idx:end_idx] = correct_batch

                state.robust_flags = robust_flags
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {'clean': robust_accuracy}
                state.clean_accuracy = robust_accuracy

                if self.verbose:
                    self.logger.log(f"initial accuracy: {robust_accuracy:.2%}")

            else:
                robust_flags = state.robust_flags.to(x_orig.device)
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {'clean': state.clean_accuracy}
                if self.verbose:
                    self.logger.log(f"initial clean accuracy: {state.clean_accuracy:.2%}")
                    self.logger.log(
                        f"robust accuracy at the time of restoring the state: {robust_accuracy:.2%}"
                    )

            x_adv = x_orig.clone().detach()
            min_margin = torch.ones(x_adv.shape[0], device=self.device)
            startt = time.time()

            for attack in attacks_to_run:
                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()
                n_batches = int(np.ceil(num_robust / bs))
                if num_robust == 0:
                    break

                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x_batch = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y_batch = y_orig[batch_datapoint_idcs].clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x_batch.shape) == 3:
                        x_batch.unsqueeze_(dim=0)

                    # run attack
                    if attack == 'apgd-ce':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'ce'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x_batch, y_batch)

                    elif attack == 'apgd-dlr':
                        # apgd on dlr loss
                        self.apgd.loss = 'dlr'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x_batch, y_batch)

                    elif attack == 'fab':
                        # fab
                        # if hasattr(self.model, 'use_nonlin_for_grad'):
                        #     self.model.use_nonlin_for_grad = False
                        self.fab.targeted = False
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x_batch, y_batch)

                    elif attack == 'square':
                        # square
                        self.square.seed = self.get_seed()
                        adv_curr = self.square.perturb(x_batch, y_batch)
                    
                    elif attack == 'apgd-t':
                        # targeted apgd
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x_batch, y_batch)

                    elif attack == 'fab-t':
                        # fab targeted
                        # if hasattr(self.model, 'use_nonlin_for_grad'):
                        #     self.model.use_nonlin_for_grad = False
                        self.fab.targeted = True
                        self.fab.n_restarts = 1
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x_batch, y_batch)

                    else:
                        raise ValueError('Attack not supported')

                    ### Update attacked images based on the margins
                    # This is different from the original implementation
                    logits_batch = self.get_logits(adv_curr).detach()
                    try:
                        probs_batch = logits_batch.softmax(dim=1)
                    except:
                        probs_batch = F.one_hot(
                            logits_batch.argmax(dim=1), num_classes=logits_batch.shape[1]
                        ).float()
                    pred_batch = logits_batch.argmax(dim=1)

                    corr_mask = (pred_batch == y_batch).detach().to(robust_flags.device)
                    incorr_mask = ~corr_mask
                    corr_local_idcs = (corr_mask).nonzero().reshape(-1)
                    incorr_local_idcs = (incorr_mask).nonzero().reshape(-1)

                    corr_global_idcs = batch_datapoint_idcs[corr_mask]
                    incorr_global_idcs = batch_datapoint_idcs[incorr_mask]
                    robust_flags[incorr_global_idcs] = False
                    state.robust_flags = robust_flags

                    # Split probs_batch into correct and incorrect
                    num_classes = probs_batch.shape[1]
                    probs_incorr = probs_batch[incorr_local_idcs].reshape(-1, num_classes)
                    probs_corr = probs_batch[corr_local_idcs].reshape(-1, num_classes)

                    # For correct examples, the margin should be positive
                    top2_corr = probs_corr.topk(k=2, dim=1).values
                    margin_corr = top2_corr[:, 0] - top2_corr[:, 1]
                    corr_update_mask = (margin_corr < min_margin[corr_global_idcs]).cpu()

                    # For incorrect examples, the margin should be negative
                    prob_incorr_gt = probs_incorr.gather(
                        dim=1, index=y_batch[incorr_local_idcs].unsqueeze(1)
                    )
                    margin_incorr = prob_incorr_gt.reshape(-1) - probs_incorr.max(dim=1)[0]
                    incorr_update_mask = (margin_incorr < min_margin[incorr_global_idcs]).cpu()

                    # Calculate the indices to update
                    global_idcs_to_update = torch.cat([
                        corr_global_idcs[corr_update_mask], incorr_global_idcs[incorr_update_mask]
                    ], dim=0)
                    local_idcs_to_update = torch.cat([
                        corr_local_idcs[corr_update_mask], incorr_local_idcs[incorr_update_mask]
                    ], dim=0)

                    # Update the margin
                    new_margin = torch.cat([
                        margin_corr[corr_update_mask], margin_incorr[incorr_update_mask]
                    ], dim=0)
                    min_margin[global_idcs_to_update] = new_margin.float()

                    # Update attacked image and class
                    x_adv[global_idcs_to_update] = adv_curr[local_idcs_to_update].cpu()
                    y_adv[global_idcs_to_update] = pred_batch[local_idcs_to_update].cpu()

                    if self.verbose: 
                        self.logger.log(
                            f"{attack} - {batch_idx + 1} / {n_batches} - {incorr_mask.sum()}"
                            f" out of {x_batch.shape[0]} successfully perturbed."
                        )

                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict[attack] = robust_accuracy
                state.add_run_attack(attack)

                if self.verbose:
                    self.logger.log(
                        f"robust accuracy after {attack.upper()}: {robust_accuracy:.2%}"
                        f"(total time {(time.time() - startt):.1f} s)"
                    )

            # check about square
            checks.check_square_sr(robust_accuracy_dict, logger=self.logger)
            state.to_disk(force=True)

            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - x_orig) ** 2).reshape(x_orig.shape[0], -1).sum(-1).sqrt()
                elif self.norm == 'L1':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).sum(dim=-1)

                self.logger.log(
                    f"max {self.norm} perturbation: {res.max():.5f}, "
                    f"nan in tensor: {(x_adv != x_adv).sum()}, "
                    f"max: {x_adv.max():.5f}, min: {x_adv.min():.5f}"
                )
                self.logger.log(f"robust accuracy: {robust_accuracy:.2%}")

        if return_labels:
            return x_adv, y_adv
        else:
            return x_adv
