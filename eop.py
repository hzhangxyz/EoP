from math import sqrt, log
from numpy.linalg import eigh, norm
from scipy.linalg import sqrtm, expm
from scipy.optimize import minimize
import torch


def get_eop(rho):
    m = rho.block[["I1", "I2", "O1", "O2"]]
    m /= norm(m)
    m = m.reshape([4, 4])
    m = sqrtm(m)
    m = torch.tensor(m)

    logU = torch.zeros_like(m, requires_grad=True, dtype=float)
    lrs = (*(1e-1 for i in range(200)), *(1e-2 for i in range(500)),
           *(1e-3 for i in range(500)), *(1e-4 for i in range(500)))
    for lr in lrs:
        U = torch.matrix_exp(logU)
        matrix = m @ U
        matrix = matrix.reshape([2, 2, 2, 2])
        matrix = matrix.transpose(2, 1)
        matrix = matrix.reshape([4, 4])
        matrix = matrix @ matrix.T
        l, v = torch.linalg.eigh(matrix)
        l = l / l.sum()
        l = torch.where(l < 0, torch.ones_like(l, dtype=float), l)
        es = -l * torch.log(l)
        e = es.sum()
        e.backward()

        with torch.no_grad():
            logU -= lr * logU.grad
            logU.grad = None
    e = float(e)
    return e
