"""
This file contains multiple method to sparsify the coefficients
"""
import time
import numpy as np
import numpy.linalg as la

# import utility
from IPython.display import (
    clear_output,
)  # This is to clean the output info to make the process cleaner

import torch
import torch.nn.functional as F

# import cupy as cp


def quadraticBasisUpdate(
    basis,
    Res,
    ahat,
    lowestActivation,
    HessianDiag,
    stepSize=0.001,
    constraint='L2',
    Noneg=False,
):
    """
    This matrix update the basis function based on the Hessian matrix of the activation.
    It's very similar to Newton method. But since the Hessian matrix of the activation function is often ill-conditioned, we takes the pseudo inverse.

    Note: currently, we can just use the inverse of the activation energy.
    A better idea for this method should be caculating the local Lipschitz constant for each of the basis.
    The stepSize should be smaller than 1.0 * min(activation) to be stable.
    """
    dBasis = stepSize * torch.mm(Res, ahat.t()) / ahat.size(1)
    dBasis = dBasis.div_(HessianDiag + lowestActivation)
    basis = basis.add_(dBasis)
    if Noneg:
        basis = basis.clamp(min=0.0)
    if constraint == 'L2':
        basis = basis.div_(basis.norm(2, 0))
    return basis


def ISTA_PN(I, basis, lambd, num_iter, eta=None, useMAGMA=True):
    # This is a positive-negative PyTorch-Ver ISTA solver
    # MAGMA uses CPU-GPU hybrid method to solve SVD problems, which is great for single task. When running multiple jobs, this flag should be turned off to leave the svd computation on only GPU.
    dtype = basis.type()
    batch_size = I.size(1)
    M = basis.size(1)
    if eta is None:
        if useMAGMA:
            L = torch.max(
                torch.linalg.eigvalsh(torch.mm(basis, basis.t()), UPLO='U')
            )
            eta = 1.0 / L
        else:
            eta = 1.0 / cp.linalg.eigvalsh(
                cp.asarray(torch.mm(basis, basis.t()).cpu().numpy())
            ).max().get().reshape(1)
            eta = torch.from_numpy(eta.astype('float32')).cuda()

    # Res = torch.zeros(I.size()).type(dtype)
    # ahat = torch.zeros(M,batch_size).type(dtype)
    Res = torch.cuda.FloatTensor(I.size()).fill_(0)
    ahat = torch.cuda.FloatTensor(M, batch_size).fill_(0)

    for t in range(num_iter):
        ahat = ahat.add(eta * basis.t().mm(Res))
        ahat_sign = torch.sign(ahat)
        ahat.abs_()
        ahat.sub_(eta * lambd).clamp_(min=0.0)
        ahat.mul_(ahat_sign)
        Res = I - torch.mm(basis, ahat)
    return ahat, Res
