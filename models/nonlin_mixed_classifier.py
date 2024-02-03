import torch
from torch import nn, Tensor

from typing import Union, List
from utils.misc_utils import outer_prod


class NonLinMixedClassifier(nn.Module):
    enable_autocast = False

    def __init__(self, std_model: nn.Module, rob_model: nn.Module, forward_settings: dict):
        """ Initialize the MixedNUTS classifier.

        Args:
            std_model (nn.Module): The standard base classifier (can be non-robust).
            rob_model (nn.Module): The robust base classifier.
            forward_settings (dict): A dictionary containing the following forward settings:
                - use_nonlin_for_grad (bool):
                    If True, use the robust base model logit nonlinearity for gradient.
                    If False, (partially) bypass the nonlinearity for
                    better gradient flow and more effective attack.
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
                  f"{sum(p.numel() for p in model.parameters())} parameters. "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} "
                  "parameters are trainable.")

        # alpha value (mixing balance)
        self.set_alpha_value(forward_settings["alpha"], forward_settings["alpha_diffable"])
        self.use_nonlin_for_grad = forward_settings["use_nonlin_for_grad"]
        print(f"{'Using' if self.use_nonlin_for_grad else 'Disabling'} "
              "robust base model nonlinear transformation for gradient calculations.") 

        # Base model output mapping functions
        self.std_map = forward_settings['std_map']
        self.rob_map = forward_settings['rob_map']
        self.softmax = nn.Softmax(dim=1)

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

    def _do_checks(self, images: Tensor):
        # Check eval mode
        for model in [self.std_model, self.rob_model]:
            assert not model.training

        # Check devices match
        if hasattr(self.std_model, "root") and (
            self.std_model.root.conv.weight.device != images.device):
            print(self.std_model.root.conv.weight.device,
                  self.rob_model.logits.weight.device, images.device)
            raise ValueError("Device mismatch!")

    def forward(self, images: Tensor, return_probs: bool = False, return_all: bool = False):
        """ Forward pass of the mixed classifier.

        Args:
            images (Tensor):
                Input images with size (n, c, h, w).
            return_probs (bool, optional):
                If True, skip and log and return the mixed probabilities.
                Otherwise, return MixedNUTS's logits. Defaults to False.
            return_all (bool, optional):
                If True, return the mixed probs/logits, the differentiable mixed probs/logits
                used for adversarial attack, and the alpha values.
                Otherwise, only return the mixed probs/logits for compatibility with existing models.
                Defaults to False.

        Returns:
            Tensor or tuple: Return values. See args for details.
        """
        self._do_checks(images)
        return_device = images.device

        # Base classifier forward passes
        with torch.cuda.amp.autocast(
            enabled=(self.enable_autocast and torch.cuda.is_available())
        ):
            # Single model special cases
            if torch.numel(self._alpha) == 1 and self._alpha.item() == 0:  # STD Model only
                logits_std = self.std_model(images)
                mapped_std = self.std_map(logits_std, return_probs=False).float()
                assert not mapped_std.isnan().any()
                alpha = torch.zeros((logits_std.shape[0],)).to(logits_std.device)
                return (mapped_std, logits_std, alpha) if return_all else mapped_std

            elif torch.numel(self._alpha) == 1 and self._alpha.item() == 1:  # ROB Model only
                logits_rob = self.rob_model(images)
                mapped_rob = self.rob_map(logits_rob, return_probs=False).float()
                assert not mapped_rob.isnan().any()
                alpha = torch.ones((logits_rob.shape[0],)).to(logits_rob.device)
                return (mapped_rob, mapped_rob, alpha) if return_all else mapped_rob

            # General case -- use both models
            logits_std = self.std_model(images)
            logits_rob = self.rob_model(images)
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

        if return_all:
            # Return mixed probs/logits, differentiable mixed probs/logits, and alphas

            if self.use_nonlin_for_grad:
                # Use the robust base model nonlinearity for gradient
                mixed_probs_diffable = outer_prod(alpha, mapped_rob) + \
                    outer_prod((1 - alpha), self.softmax(logits_std).to(mapped_std.device))
                mixed_logits_diffable = torch.log(mixed_probs_diffable)
            else:
                # Disable the robust base model nonlinearity for gradient (usually stronger)
                raw_ratio = .9
                probs_std = self.softmax(logits_std).to(mapped_std.device)
                probs_rob = self.softmax(logits_rob).to(mapped_rob.device)
                mixed_probs_diffable = \
                    outer_prod((1 - alphas_diffable), probs_std) + \
                    outer_prod(alphas_diffable * raw_ratio, probs_rob) + \
                    outer_prod(alphas_diffable * (1 - raw_ratio), mapped_rob)
                mixed_logits_diffable = torch.log(mixed_probs_diffable)

            assert not mixed_logits_diffable.isnan().any()
            if return_probs:
                return (
                    mixed_probs.float().to(return_device),
                    mixed_probs_diffable.float().to(return_device),
                    alpha.reshape(-1).to(return_device)
                )
            else:
                return (
                    mixed_logits.float().to(return_device),
                    mixed_logits_diffable.float().to(return_device),
                    alpha.reshape(-1).to(return_device)
                )

        else:
            # Only return the mixed probs/logits
            return mixed_probs.float().to(return_device) if return_probs \
                else mixed_logits.float().to(return_device)
