import torch
from torch import nn, Tensor

from typing import Union, List
from utils.misc_utils import outer_prod


class NonLinMixedClassifier(nn.Module):

    def __init__(self, std_model: nn.Module, rob_model: nn.Module, forward_settings: dict):
        """ Initialize the MixedNUTS classifier.

        Args:
            std_model (nn.Module): The standard base classifier (can be non-robust).
            rob_model (nn.Module): The robust base classifier.
            forward_settings (dict): A dictionary containing the following forward settings:
                - std_map (nn.Module):
                    The mapping function for the standard base model logits.
                - rob_map (nn.Module):
                    The mapping function for the robust base model logits.
                - alpha (float or Tensor):
                    The mixing weight between the two base classifiers.
                    If a float, the MixedNUTS output has shape (n, num_classes).
                    If a Tensor with shape (m1, m2, ..., md), the MixedNUTS output has shape
                    (n, num_classes, m1, ..., md). I.e., an outer product is performed.
                - alpha_diffable (float or Tensor):
                    The mixing weight between the two base classifiers for differentiable output.
                    The shape-dependent behavior is the same as alpha.
        """
        super().__init__()
        self.std_model, self.rob_model = std_model, rob_model

        # Freeze models and set to eval mode
        for model, name in zip([self.std_model, self.rob_model], ["STD", "ROB"]):
            model.eval()
            for param in model.parameters():
                assert param.requires_grad == False
            print(f"The {name} classifier has "
                  f"{sum(param.numel() for param in model.parameters())} parameters.")

        # alpha value (mixing balance)
        self.set_alpha_value(forward_settings["alpha"], forward_settings["alpha_diffable"])

        # Base model logit transformations
        self.std_map, self.rob_map = forward_settings['std_map'], forward_settings['rob_map']

    def set_alpha_value(
        self, alpha: Union[float, int, Tensor, List],
        alpha_diffable: Union[float, int, Tensor, List], verbose=True
    ):
        """ Set the alpha value for the mixed classifier. """
        # Set alpha
        self._alpha = nn.parameter.Parameter(torch.tensor(alpha).float(), requires_grad=False)
        assert self._alpha.min() >= 0 and self._alpha.max() <= 1, \
            "The range of alpha should be [0, 1]."
        if verbose:
            print(f"Using alpha={alpha}.")
        if torch.numel(self._alpha) == 1 and self._alpha.item() == 0:
            print("Using the STD network only.")
        elif torch.numel(self._alpha) == 1 and self._alpha.item() == 1:
            print("Using the ROB network only.")

        # Set alpha_diffable
        self._alpha_diffable = nn.parameter.Parameter(
            torch.tensor(alpha_diffable).float(), requires_grad=False
        )
        assert self._alpha_diffable.min() >= 0 and self._alpha_diffable.max() <= 1, \
            "The range of alpha_diffable should be [0, 1]."
        if verbose and not torch.allclose(self._alpha_diffable, self._alpha, rtol=1e-8):
            print(f"Using alpha_diffable={alpha_diffable}.")
        if torch.numel(self._alpha_diffable) == 1 and self._alpha_diffable.item() == 0:
            print("Using the STD network only for differentiable output.")
        elif torch.numel(self._alpha_diffable) == 1 and self._alpha_diffable.item() == 1:
            print("Using the ROB network only for differentiable output.")

        assert self._alpha.shape == self._alpha_diffable.shape, \
            "alpha and alpha_diffable must have the same shape."

    def forward(self, images: Tensor, return_probs: bool = False, grad_bypass: bool = True):
        """ Forward pass of the mixed classifier.

        Args:
            images (Tensor):
                Input images with size (n, c, h, w).
            return_probs (bool, optional):
                If True, skip and log and return the mixed probabilities.
                Otherwise, return MixedNUTS's logits. Defaults to False.
            grad_bypass (bool, optional):
                If True, (mostly) bypass the nonlinear logit transformations when building
                the gradient graph. This makes MixedNUTS compatible with gradient-based
                attacks, enabling AutoAttack to reliably evaluate its robustness.
                Otherwise, include the logit transformations in gradient calculation.
                Defaults to False.

        Returns:
            Tensor:
                The mixed logits or probabilities (depending on the value of return_probs).
        """
        assert not self.std_model.training, "The accurate base classifier should be in eval mode."
        assert not self.rob_model.training, "The robust base classifier should be in eval mode."
        return_device = images.device
        raw_ratio = .9  # The ratio of the raw robust base model output to use for gradient

        # Base classifier forward passes
        # Accurate base classifier only
        if torch.numel(self._alpha) == 1 and self._alpha.item() == 0:
            logits_std = self.std_model(images)
            raw_std = logits_std.softmax(dim=1) if return_probs else logits_std
            mapped_std = self.std_map(logits_std, return_probs=return_probs).float()
            assert not mapped_std.isnan().any()
            if grad_bypass:
                output = raw_std + (mapped_std - raw_std).detach()
            return output

        # Robust base classifier only
        elif torch.numel(self._alpha) == 1 and self._alpha.item() == 1:
            logits_rob = self.rob_model(images)
            raw_rob = logits_rob.softmax(dim=1) if return_probs else logits_rob
            mapped_rob = self.rob_map(logits_rob, return_probs=return_probs).float()
            assert not mapped_rob.isnan().any()
            if grad_bypass:
                output = (1 - raw_ratio) * mapped_rob + \
                    raw_ratio * (raw_rob + (mapped_rob - raw_rob).detach())
            return output

        # General case -- use both base classifiers
        logits_std, logits_rob = self.std_model(images), self.rob_model(images)
        assert logits_std.device == logits_rob.device

        # Apply nonlinear logit transformations and convert to probabilities
        mapped_std = self.std_map(logits_std, return_probs=True)
        mapped_rob = self.rob_map(logits_rob, return_probs=True)

        alpha = self._alpha.to(mapped_rob.device)
        alphas_diffable = self._alpha_diffable.to(mapped_rob.device)

        # Mix the output probabilities of the two base classifiers
        mixed_probs = outer_prod((1 - alpha), mapped_std) + outer_prod(alpha, mapped_rob)
        # Log is the inverse of the softmax
        mixed_logits = torch.log(mixed_probs)
        assert not mixed_logits.isnan().any()

        # (mostly) Bypass the base model nonlinearities for gradient calculations
        # Enable AutoAttack to reliably evaluate the robustness of the mixed classifier
        if grad_bypass:
            probs_std = logits_std.softmax(dim=1).to(mapped_std.device)
            probs_rob = logits_rob.softmax(dim=1).to(mapped_rob.device)
            mixed_probs_diffable = \
                outer_prod((1 - alphas_diffable), probs_std) + \
                outer_prod(alphas_diffable * raw_ratio, probs_rob) + \
                outer_prod(alphas_diffable * (1 - raw_ratio), mapped_rob)
            mixed_logits_diffable = torch.log(mixed_probs_diffable)
            assert not mixed_logits_diffable.isnan().any()

            # Preserve the mixing results with nonlinear logit transformations
            # while using the mixture without the transformations in the gradient graph
            mixed_probs = mixed_probs_diffable + \
                (mixed_probs - mixed_probs_diffable).detach()
            mixed_logits = mixed_logits_diffable + \
                (mixed_logits - mixed_logits_diffable).detach()

        if return_probs:
            return mixed_probs.float().to(return_device)
        else:
            return mixed_logits.float().to(return_device)
