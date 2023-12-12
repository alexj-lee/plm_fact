import torch
from torch import nn
from torch.nn import functional as F
from plm_fact.models.sparse_code._external.sparsify_PyTorch import (
    quadraticBasisUpdate,
    ISTA_PN,
)
from typing import Optional


class SparseCoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        basis_dim: int,
        num_steps_avg: int,
        lamb: Optional[float] = 0.3,
        n_iters: Optional[int] = 500,  # from zeyu paper
        h: Optional[float] = 0.005,
    ):
        super().__init__()

        basis = torch.randn(embed_dim, basis_dim)
        basis = basis.div_(basis.norm(2, 0))

        self.basis = nn.Parameter(basis)

        self.hessian_diag_approx = nn.Parameter(torch.zeros(basis_dim))

        self.history_len = num_steps_avg
        if not (num_steps_avg > 0):
            raise ValueError(
                f"num_steps_avg must be a positive integer. You passed {num_steps_avg}"
            )
        self.running_mean_const = (self.history_len - 1) / self.history_len

        l1_history = torch.Tensor(basis_dim).float()
        self.register_buffer('l1_history', l1_history)

        signal_energy = torch.Tensor(0).float()
        noise_energy = torch.Tensor(0).float()

        self.register_buffer(
            'signal_energy',
            signal_energy,
        )

        self.register_buffer('noise_energy', noise_energy)

        eye = torch.zeros(embed_dim, basis_dim, dtype=torch.float32)
        self.register_buffer('eye', eye)

        self.lamb = lamb
        self.n_iters = n_iters

        self.h = h

    def _update(self, tens: torch.Tensor):
        # running average updates
        return tens * self.running_mean_const

    def forward(self, I):
        # I should be (dim, bs)
        I = I.T

        a_hat, res = ISTA_PN(I, self.basis, self.lamb, self.n_iters)

        self.l1_history = (
            self._update(self.l1_history)
            + a_hat.abs().mean(1) / self.history_len
        )

        self.hessian_diag_approx = (
            self._update(self.hessian_diag_approx)
        ) + a_hat.pow(2).mean(1) / self.history_len

        signal_energy = (
            self._update(self.signal_energy)
            + I.pow(2).sum() / self.history_len
        )

        noise_energy = self._update(self.noise_energy)

        snr = signal_energy / noise_energy

        self.basis = quadraticBasisUpdate(
            self.basis,
            res,
            a_hat,
            0.001,  # not sure why this is hardcoded to 0.001 for both Zeyu and Yubei code
            self.hessian_diag_approx,
            self.h,
        )

        return dict(
            snr=snr,
            hes_min=self.hessian_diag_approx.min(),
            hes_max=self.hessian_diag_approx.max(),
            l1_min=self.l1_history.min(),
            l1_max=self.l1_history.max(),
        )
